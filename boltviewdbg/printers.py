"""Module containing pretty printers for gdb for BoltView

Use register_printers(obj) function to register printers for 'obj' obj file.

Example::

	import sys
	import gdb

	sys.path.insert(0, '..')

	import boltviewdbg.printers

	boltviewdbg.printers.register_printers(gdb.current_objfile())

"""

from __future__ import absolute_import, division, print_function, unicode_literals

__copyright__ = 'Copyright 2017 Eyen SE'
__author__ =  'Jakub Klener jakub.klener@eyen.eu'

import copy
import gdb
import re
import sys


### Python 2 compatibility
if sys.version_info[0] > 2:
	### Python 3
	Iterator = object
	# Also, int subsumes long
	long = int

	def iterate_items(to_iterate):
		return to_iterate.items()
else:
	### Python 2 stuff
	class Iterator(object):
		def next(self):
			return self.__next__()

	# items() method is slow in python2
	def iterate_items(to_iterate):
		return to_iterate.iteritems()

# Try to use the new-style pretty-printing if available.
_use_gdb_pp = True
try:
	import gdb.printing
except ImportError:
	_use_gdb_pp = False


# Try to install type-printers.
_use_type_printing = False
try:
	import gdb.types
	if hasattr(gdb.types, 'TypePrinter'):
		_use_type_printing = True
except ImportError:
	pass


def get_basic_typename(valtype):
	"""Gets original type name.

	Args:
		valtype (gdb.Type): Type to be converted.

	Returns:
		str: stripped from references and and all typedefs.

	"""
	valtype = gdb.types.get_basic_type(valtype)
	return valtype.tag


def is_specialization_of(typename, template_name):
	"""Tests if template from std library is specialization of another template.

	Args:
		typename (str):	Full name of specialized template without angle brackets.
		template_name (str): Name of template without angle brackets.

	Returns:
		str: stripped from references and and all typedefs.

	"""
	return re.match('^std::.*%s(<.*>)$' % (template_name, ), typename) is not None


# inspired by libstc++ pretty printers
def get_unique_ptr_target(ptr):
	"""Returns unique_ptr target despite of new or old implementation.

	Args:
		ptr(gdb.Value): std::unique_ptr

	Returns:
		gdb.Value: Raw pointer

	"""
	impl_type = ptr['_M_t'].type.fields()[0].type.tag
	if is_specialization_of(impl_type, '__uniq_ptr_impl'): # New implementation
		v = ptr['_M_t']['_M_t']['_M_head_impl']
	elif is_specialization_of(impl_type, '_Tuple_impl'):
		v = ptr['_M_t']['_M_head_impl']
	else:
		raise ValueError('Unsupported implementation for unique_ptr: %s' % ptr.type.fields()[0].type.tag)
	return v


class RecognizeTypeMixin(object):
	"""This mixin enables type recognizing in the child class through the
	recognize_type method.
	"""
	def recognize_type(self, valtype):
		"""Returns typename if type recognizers are installed.
		Args:
			valtype (gdb.Type)

		Returns:
			str

		"""
		global _use_type_printing
		if not _use_type_printing:
			return get_basic_typename(valtype)
		return gdb.types.apply_type_recognizers(gdb.types.get_type_recognizers(),
												gdb.types.get_basic_type(valtype)) or get_basic_typename(valtype)


class SimpleDataPrinter(object):
	"""Prints data from an object."""
	def __init__(self, value, prefix, typename, membername):
		"""Args:
			value (gdb.Value): represents object to be printed
			prefix (str): prefix for the purposes of the object name printing.
				For	example `bolt::`
			typename (str): typename for the purposes of the object name
				printing.
			membername (str): name of an object member to be printed.

		"""
		self.out = value[membername]

	def to_string(self):
		return self.out


class MemberPrinter(object):
	"""Prints data from an object as children.

	Derived classes can modify `children_list` and `childdata` variables. The
	`children_list` specifies which of and in which order would be childdata
	printed. The `childdata` is dictionary containing objects convertible to str
	(e.g. gdb.Value or str). These data would be printed as children of the
	object (enclsed in curly braces).

	The `to_string` method can be also reimplemented to modify name of printed
	object. No name is printed if it returns `None`.

	"""
	def __init__(self, value, children = None, childdata = None):
		"""Args:
			value (gdb.Value): represents object to be printed
			children (List[str]): Names of data members to be printed. Defines
				order of printing. These names don't have to be same as data
				member names but has to match keys from `childdata`.
			childdata (Dict[str, str]): Defines data, to be printed. Keys are
				mathced with values stored in `children` list and values is
				names of data members to be printed.

		"""
		self.value = value

		# default arguments for mutable types points to the same object so we
		# have to initialize them to None
		if children is None:
			self.children_list = []
		else:
			self.children_list = children

		if childdata is None:
			self.childdata = {}
		else:
			self.childdata = {k: value[v] for k, v in iterate_items(childdata)}

	def children(self):
		"""Iterates through `children_list` and	prints corresponding data from
		`childdata`.
		"""
		# return iterable of tuples as required by pretty printing API
		return [(key, self.childdata[key]) for key in self.children_list]

	def to_string(self):
		"""Prints a name of the object. Prints nothing if the return value is
		`None`.
		"""
		return None


class BoltFunctorMemberPrinter(MemberPrinter):
	"""General printer for bolt objects.

	It prints bolt object name and members specified in `children` and
	`childdata` variables, see `MemberPrinter`.

	"""
	def __init__(self, value, prefix, typename, *args, **kwargs):
		"""Args:
			prefix (str): prefix for the purposes of the object name printing.
				For	example `bolt::`
			typename (str): typename for the purposes of the object name
				printing.
			value (gdb.Value): gdb representation of object to be printed
			Other arguments are passed to the parent class - see
				`MemberPrinter` for more info.

		"""
		super(BoltFunctorMemberPrinter, self).__init__(value, *args, **kwargs)
		self.typename = typename

	def to_string(self):
		"""Prints a name of the object. Prints nothing if the return value is
		`None`.
		"""
		return self.typename


class BoltMemberPrinter(MemberPrinter):
	"""General printer for bolt objects.

	It prints bolt object name and members specified in `children` and
	`childdata` variables, see `MemberPrinter`.

	"""
	def __init__(self, value, prefix, typename, *args, **kwargs):
		"""Args:
			prefix (str): prefix for the purposes of the object name printing.
				For	example `bolt::`
			typename (str): typename for the purposes of the object name
				printing.
			value (gdb.Value): gdb representation of object to be printed
			Other arguments are passed to the parent class - see
				`MemberPrinter` for more info.

		"""
		super(BoltMemberPrinter, self).__init__(value, *args, **kwargs)

	def to_string(self):
		"""Prints a name of the object. Prints nothing if the return value is
		`None`.
		"""
		return None


class BoltImagePrinter(BoltMemberPrinter):
	"""General printer for bolt images.

	It prints `size_` and `strides_` data members.

	"""
	def __init__(self, value, *args, **kwargs):
		"""Args:
			value (gdb.Value): gdb representation of object to be printed
			Other arguments are passed to the parent class - see
				`BoltMemberPrinter` for more info.

		"""
		super(BoltImagePrinter, self).__init__(value, *args, **kwargs)
		self.children_list.extend(['size', 'strides']),
		self.childdata.update({
			'size': value['size_'],
			'strides': value['strides_']
		})


class GetImageDataAddressesMixin(object):
	"""Adds method `_get_data_addresses` which adds addresses of beginning and
	one past end of the array with data as 'data' member of the `childdata`
	dict (see `MemberPrinter`).

	In the derived class, you have to define
	`_get_ptr(value: gdb.Value, ptr_name: str) -> gdb.Value`. Return type has
	to represent raw pointer.

	The printed type have to contain `size_` and `strides_` data members.

	Example::
		class DataImagePrinter(GetImageDataAddressesMixin, MemberPrinter):
			def __init__(self, value, *args, **kwargs):
				super(MemberPrinter, self).__init__(value, *args, **kwargs)
				# addresses will be saved as "data" member
				self.children_list.extend(['data'])
				self._get_data_addresses(value, 'ptr_')

			# _get_ptr() method has to be defined
			def _get_ptr(self, value, ptr_name):
				return value[ptr_name]

	"""
	def _get_dimension(self, value):
		"""Find dimension from template name because it can be optimized out."""
		# TODO(jakub): Implement it as xmethod
		typename = get_basic_typename(value.type)
		match = re.match(r'^[a-zA-Z0-9_:]+<.*([1-9]+)>$', typename)
		return int(match.group(1))

	def _get_data_addresses(self, value, ptr_name):
		"""Adds `data` member with addresses of beginning and one past end to
		the childdata dict.

		It requires `_get_ptr()` method, to be defined. See class description.

		Args:
			value (gdb.Value): Image or View.
			ptr_name (str): name of member pointing to data.

		"""
		sizedata = value['size_']['data']
		stridedata = value['strides_']['data']

		# compute occupied memory size
		memsize = 0
		for i in range(self._get_dimension(value)):
			memsize += stridedata[i] * (sizedata[i] - 1)

		try:
			start_ptr = self._get_ptr(value, ptr_name)
		except AttributeError:
			raise AttributeError('GetDataAdressesMixin: You have to define _get_ptr method which returns data address.')
		end_ptr = start_ptr + memsize + start_ptr.dereference().type.sizeof

		self.childdata['data'] = '%s:%s' % (start_ptr, end_ptr)


class BoltDataImagePrinter(GetImageDataAddressesMixin, BoltImagePrinter):
	"""General printer for bolt images with standard memory layout.

	It requires for its children to define _get_ptr() method, see
	`GetImageDataAddressesMixin`.

	"""
	def __init__(self, value, prefix, typename, *args, **kwargs):
		"""Args:
			prefix (str): prefix for the purposes of the object name printing.
				For	example `bolt::`
			typename (str): typename for the purposes of the object name
				printing.
			value (gdb.Value): gdb representation of object to be printed
			Other arguments are passed to the ancestors.

		Kwargs:
			ptr_name (str): name of pointer to data.
			Other kwargs are passed to ancestors.

		"""
		try:
			ptr_name = kwargs.pop('ptr_name')
		except KeyError:
			raise TypeError('BoltDataImagePrinter: You have to provide ptr_name keyword argument.')

		super(BoltDataImagePrinter, self).__init__(value, prefix, typename, *args, **kwargs)
		self.children_list.extend(['data'])
		self._get_data_addresses(value, ptr_name)


class BoltRawImagePrinter(BoltDataImagePrinter):
	"""Prints an Image or View object with raw pointer to data."""
	def __init__(self, *args, **kwargs):
		"""Args:
			Arguments are passed to ancestors.

		"""
		super(BoltRawImagePrinter, self).__init__(*args, **kwargs)

	def _get_ptr(self, value, ptr_name):
		"""Definition of method, which returns raw pointer to the stored image.
		see GetImageDataAddressesMixin.
		"""
		return value[ptr_name]


class BoltUniqueImagePrinter(BoltDataImagePrinter):
	"""Prints a HostImage object."""
	def __init__(self, *args, **kwargs):
		"""Args:
			Arguments are passed to ancestors.

		"""
		super(BoltUniqueImagePrinter, self).__init__(*args, **kwargs)

	def _get_ptr(self, value, ptr_name):
		"""Definition of method, which returns raw pointer to the stored image.
		see GetImageDataAddressesMixin.
		"""
		unique_ptr = value[ptr_name]
		return get_unique_ptr_target(unique_ptr)


class BoltTextureImagePrinter(BoltImagePrinter):
	"""Prints a TextureImage object."""
	#TODO(jakub): investigate texture memory layout.
	def __init__(self, value, prefix, typename, *args, **kwargs):
		"""Args:
			value (gdb.Value): gdb representation of object to be printed
			prefix (str): prefix for the purposes of the object name printing.
				For	example `bolt::`
			typename (str): typename for the purposes of the object name
				printing.
			Other arguments are passed to ancestors.


		Kwargs:
			texture_object (str): name texture object identifier.
			Other kwargs are passed to ancestors.

		"""
		try:
			texture_object = kwargs.pop('texture_object')
		except KeyError:
			raise TypeError('BoltTextureImagePrinter: You have to provide texture_object keyword argument.')

		super(BoltTextureImagePrinter, self).__init__(value, prefix, typename, *args, **kwargs)
		self.children_list.extend(['cudaArray', 'cudaTextureObject']),
		self.childdata.update({
			'cudaArray': value['cuda_array_'],
			'cudaTextureObject': value[texture_object]
		})


class Subprinter(object):
	"""Adapter for printer classes, which uses API of `Printer` class callable
	for	registering pretty printers.

	Each time a variable with the matching type is inspected, the `Printer`
	calls `invoke` method.

	"""
	def __init__(self, name, printer, *args, **kwargs):
		"""Args:
			name (str): name displayed if `info pretty-printer is invoked`
			printer (pretty-printable object): A callable the instance of which
				is convertible to `str`. It can contain `children()` method
				which returns iterable of tuples [(key, child)] which are
				recursively	pretty printed too. It takes `value` as it's first
				argument
			Other arguments are passed to `printer`.

		"""
		self.name = name
		self.printer = printer
		self.args = args
		self.kwargs = kwargs
		self.enabled = True

	def invoke(self, value):
		"""Called when a variable with the matching type is inspected.

		Args:
			value (gdb.Value): representation of object to be printed

		Returns:
			pretty-printable object, see the constructor comment.
		"""
		if not self.enabled:
			return None

		if value.type.code == gdb.TYPE_CODE_REF:
			if hasattr(gdb.Value, 'referenced_value'):
				value = value.referenced_value()
		return self.printer(value, *self.args, **self.kwargs)


class Printer(object):
	"""A pretty-printer object which registers its subprinters.

	Each time some data is pretty-printed the Printer object is called
	(__call__ method). This method returns pretty-printable object (see
	`Subprinter`) or None if there isn't any pretty-printer for the required
	type.

	All subprinters have to be saved in subprinters member list and can be
	listed by the `info pretty-printer` command. This command also displays
	the name of the printer stored in `name` member.

	The printer does not work, if the enable attribute is not set to True.

	"""
	def __init__(self, name):
		"""Args:
			name (str): The printer name displayed in gdb.

		"""
		super(Printer, self).__init__()
		self.name = name
		self.subprinters = []
		self.lookup = {}
		self.enabled = True
		self.template_regex = re.compile('^([a-zA-Z0-9_:]+)(<.*>)?$')

	def add(self, printer, prefix, typename, *args, **kwargs):
		"""Adds subprinter for type `prefix + typename`.

		Subprinters are invoked in reversal order of their registration (The
		newest the first), so register the least specialized first and then
		the more specialized printers.

		Args:
			printer (pretty-printable object): object passed to `Subprinter`.
				See `Subprinter`.
			prefix (str): A prefix which is used for type lookup and value
				printing.
			typename (str): Name of type for which the printer is registered.
				It is used in the combination with `prefix`. It is also used
				for type printing and for Subprinter name.
			Other arguments are passed to Subprinter.
		"""
		# A small sanity check.
		if not self.template_regex.match(prefix + typename):
			raise ValueError('programming error: "%s" does not match' % typename)

		printer = Subprinter(typename, printer, prefix, typename, *args, **kwargs)
		self.subprinters.append(printer)
		self.lookup[prefix + typename] = printer

	def __call__(self, val):
		"""Called each time some data is pretty-printed. If it returns None,
		the lookup continues, if it returns pretty-printable object, the
		object is used for pretty-printing, see `Subprinter`

		It tests, if type of `val` matches some of registered subprinters
		(which are stored in `lookup` dict). If it is, the subprinter's `invoke`
		method is called.
		"""
		typename = get_basic_typename(val.type)
		if not typename:
			return None

		# Templates.
		match = self.template_regex.match(typename)
		if not match:
			return None

		basename = match.group(1)

		if val.type.code == gdb.TYPE_CODE_REF:
			if hasattr(gdb.Value, 'referenced_value'):
				val = val.referenced_value()

		if basename in self.lookup:
			return self.lookup[basename].invoke(val)

		# Cannot find a pretty printer.  Return None.
		return None


class TypePrinter(object):
	"""
	A type printer for classes and typedefs.

	Each type printer has to define `name` and `enabled` members. It also must
	have `instantiate` method which returns `None` or recognizer class with
	`recognize` method defined. The method returs object convertible to string
	which is printed.

	"""
	def __init__(self, prefix, match, name):
		"""Args:
			prefix (str): prefix used for lookup and type printing
			match (str): type name without prefix. It is tested, if this
				type name with prefix is in the printed type name.
			name (str): The name, which is printed in the case of matching type.

		"""
		self.prefix = prefix
		self.match = match

		# following variables has to be defined in gdb type printers
		self.name = name
		self.enabled = True

	class _recognizer(object):
		"""Recognizer class, see `TypePrinter` description."""
		def __init__(self, prefix, match, name):
			"""Args:
				prefix (str): prefix used for lookup and type printing
				match (str): type name without prefix. It is tested, if this
					type name with prefix is in the printed type name.
				name (str): The name, which is printed in the case of matching
					type.

			"""
			self.prefix = prefix
			self.match = self.prefix + match
			self.name = name
			self.type_obj = None

		def recognize(self, type_obj):
			"""Method, which is used for type recognition, see `TypePrinter`
			description.

			Args:
				type_obj (gdb.Type): searched type

			Returns:
				None or to-str-convertible object
			"""
			if type_obj.tag is None:
				return None

			if self.type_obj is None:
				if not self.match in type_obj.tag:
					# Filter didn't match.
					return None

				try:
					self.type_obj = gdb.lookup_type(self.prefix + self.name).strip_typedefs()
				except gdb.error:
					pass
			if self.type_obj == type_obj:
				return self.prefix + self.name
			return None

	def instantiate(self):
		"""This method is called during type printer registration.

		Returns:
			recognizer: see this class description.
		"""
		return self._recognizer(self.prefix, self.match, self.name)


class TemplateTypePrinter(object):
	"""A type printer for class templates.

	Each type printer has to define `name` and `enabled` members. It also must
	have `instantiate` method which returns `None` or recognizer class with
	`recognize` method defined. The method returs object convertible to string
	which is printed.

	"""

	def __init__(self, prefix, typename):
		"""Args:
			prefix (str): prefix used for lookup and type printing
			typename (str): The name, which is used in type matching. It is
				without template parameters and angle brackets.

		"""
		self.prefix = prefix

		# following variables has to be defined in gdb type printers
		self.name = typename
		self.enabled = True

	class _recognizer(object):
		"""Recognizer class, see `TemplateTypePrinter` description."""
		def __init__(self, prefix, name):
			"""Args:
				prefix (str): prefix used for lookup and type printing
				name (str): The name, which is used in type matching. It is
					without template parameters and angle brackets.

			"""
			self.prefix = prefix
			self.name = name
			self.type_obj = None
			self.template_regex = re.compile('^' + prefix + '([a-zA-Z0-9_:]+)(<.*>)$')

		def recognize(self, type_obj):
			"""Method, which is used for type recognition, see
			`TemplateTypePrinter` description.

			Args:
				type_obj (gdb.Type): searched type

			Returns:
				None or to-str-convertible object
			"""
			typename = get_basic_typename(type_obj)
			if not typename:
				return None

			match = self.template_regex.match(typename)
			if not match:
				return None

			basename = match.group(1)
			if basename == self.name:
				return basename + match.group(2)
			return None

	def instantiate(self):
		"""This method is called during type printer registration.

		Returns:
			recognizer: see this class description.
		"""
		return self._recognizer(self.prefix, self.name)


class TemplateDefaultTypePrinter(object):
	r"""
	A type printer for class templates with default type parameters.

	Recognizes type names that match a regular expression.
	Replaces them with a formatted string which can use replacement field
	{N} to refer to the \N subgroup of the regex match.
	Type printers are recusively applied to the subgroups. If you don't
	want to apply printer to subgroup, for example because it is
	non-type parameter, use {!s} syntax.

	This allows recognizing e.g.
	"bolt::TextureImage<(.*), (.*), bolt::CudaType<\\1> >"
	and replacing it with "bolt::TextureImage<{1}, {2!s}>",
	omitting the template argument that uses the default type.

	Each type printer has to define `name` and `enabled` members. It also must
	have `instantiate` method which returns `None` or recognizer class with
	`recognize` method defined. The method returs object convertible to string
	which is printed.
	"""

	def __init__(self, prefix, name, pattern, subst):
		"""Args:
			prefix (str): prefix used for lookup and type printing
			name (str): The name, which is used as type printer name
			pattern (str): regular expression for type recognition and reuse
				of parameters. See the class description.
			subst (str): python formated string for template type
				simplification. See the class description.

		"""
		self.prefix = prefix
		self.pattern = re.compile('^' + prefix + pattern)
		self.subst = subst

		# following variables has to be defined in gdb type printers
		self.name = name
		self.enabled = True

	class _recognizer(RecognizeTypeMixin):
		"""Recognizer class, see `TemplateDefaultTypePrinter` description."""
		def __init__(self, prefix, name, pattern, subst):
			"""Args:
				prefix (str): prefix used for lookup and type printing
				name (str): The name, which is used as type printer name
				pattern (str): regular expression for type recognition and reuse
					of parameters. See the class `TemplateDefaultTypePrinter`
					description.
				subst (str): python formated string for template type
					simplification. See the `TemplateDefaultTypePrinter`
					description.

			"""
			self.prefix = prefix
			self.name = name
			self.pattern = pattern
			self.subst = subst
			self.type_obj = None

		def recognize(self, type_obj):
			"""Method, which is used for type recognition, see
			`TemplateDefaultTypePrinter` description.

			Args:
				type_obj (gdb.Type): searched type

			Returns:
				None or to-str-convertible object
			"""
			if type_obj.tag is None:
				return None

			m = self.pattern.match(type_obj.tag)
			if m:
				subs = list(m.groups())
				for i, sub in enumerate(subs):
					if ('{%d}' % (i+1)) in self.subst:
						# apply recognizers to subgroup
						try:
							valtype = gdb.lookup_type(sub)
						except gdb.error:
							rep = sub
						else:
							rep = self.recognize_type(valtype)
						if rep:
							subs[i] = rep
				subs = [None] + subs
				return self.subst.format(*subs)
			return None

	def instantiate(self):
		"""This method is called during type printer registration.

		Returns:
			recognizer: see this class description.
		"""
		return self._recognizer(self.prefix, self.name, self.pattern, self.subst)


def _add_typedef_printer(obj, prefix, match, name):
	"""Registeres type printer for typedefs and basic types.

	Args:
		prefix (str): prefix, for example "bolt::".
		match (str): type name to match in type lookup.
		name (str): type name to print.
	"""
	printer = TypePrinter(prefix, match, name)
	gdb.types.register_type_printer(obj, printer)


def _add_template_type_printer(obj, prefix, name):
	"""Registeres type printer for templates.

	Args:
		prefix (str): prefix, for example "bolt::".
		name (str): type name to match without template parameters and angle
			brackets.
	"""
	printer = TemplateTypePrinter(prefix, name)
	gdb.types.register_type_printer(obj, printer)


def _add_template_default_type_printer(obj, prefix, name, matches):
	"""Registeres type printer for template with default parameters.

	Args:
		prefix (str): prefix, for example "bolt::".
		name (str): type name to match for full template without default
		 	template parameters, without template parameters and angle
			brackets.
		matches (List[Tuple(str, str)]): list of matches for specialized
			templates or templates with default parameters. The less
			specialized should be the first. First member of tuple is
			regular expression to match, the second python formated string.
			See TemplateDefaultTypePrinter description for more details.

	"""

	# printer has to be added first before default type printers
	printer = TemplateTypePrinter(prefix, name)
	gdb.types.register_type_printer(obj, printer)
	for match, subst in matches:
		default_type_printer = TemplateDefaultTypePrinter(prefix, name, match, subst)
		gdb.types.register_type_printer(obj, default_type_printer)


def _register_type_printers(obj):
	"""Function used to type printer registration."""
	global _use_type_printing

	if not _use_type_printing:
		return

	prefix = 'bolt::'

	# hybrid vector
	_add_typedef_printer(obj, prefix, 'Vector<int, 2>', 'Int2')
	_add_typedef_printer(obj, prefix, 'Vector<int, 3>', 'Int3')
	_add_typedef_printer(obj, prefix, 'Vector<int, 4>', 'Int4')
	_add_typedef_printer(obj, prefix, 'Vector<bool, 2>', 'Bool2')
	_add_typedef_printer(obj, prefix, 'Vector<bool, 3>', 'Bool3')
	_add_typedef_printer(obj, prefix, 'Vector<bool, 4>', 'Bool4')
	_add_typedef_printer(obj, prefix, 'Vector<float, 2>', 'Float2')
	_add_typedef_printer(obj, prefix, 'Vector<float, 3>', 'Float3')
	_add_typedef_printer(obj, prefix, 'Vector<float, 4>', 'Float4')

	# images
	_add_template_type_printer(obj, prefix, 'DeviceImage')
	_add_template_type_printer(obj, prefix, 'HostImage')
	_add_template_type_printer(obj, prefix, 'UnifiedImage')
	_add_template_type_printer(obj, prefix, 'TextureImage')
	_add_template_default_type_printer(obj, prefix, 'TextureImage',
			[('TextureImage<(.*), (.*), bolt::CudaType<\\1 ?> >',
					'TextureImage<{1}, {2!s}>')])

	# views
	_add_template_type_printer(obj, prefix, 'DeviceImageView')
	_add_template_type_printer(obj, prefix, 'DeviceImageConstView')
	_add_template_type_printer(obj, prefix, 'HostImageView')
	_add_template_type_printer(obj, prefix, 'HostImageConstView')
	_add_template_type_printer(obj, prefix, 'UnifiedImageView')
	_add_template_type_printer(obj, prefix, 'UnifiedImageConstView')
	_add_template_type_printer(obj, prefix, 'TextureImageView')
	_add_template_type_printer(obj, prefix, 'TextureImageConstView')

	# procedural views
	_add_template_type_printer(obj, prefix, 'ConstantImageView')
	_add_template_type_printer(obj, prefix, 'CheckerBoardImageView')
	_add_template_type_printer(obj, prefix, 'BinaryOperatorImageView')
	_add_template_type_printer(obj, prefix, 'LinearCombinationImageView')
	_add_template_type_printer(obj, prefix, 'MultiplicationImageView')
	_add_template_type_printer(obj, prefix, 'DivisionImageView')
	_add_template_type_printer(obj, prefix, 'MirrorImageView')
	_add_template_type_printer(obj, prefix, 'PaddedImageView')
	_add_template_type_printer(obj, prefix, 'UnaryOperatorImageView')
	_add_template_type_printer(obj, prefix, 'MeshGridView')

	# functors
	_add_template_type_printer(obj, prefix, 'IdentityFunctor')
	_add_template_type_printer(obj, prefix, 'SquareFunctor')
	_add_template_type_printer(obj, prefix, 'SquareRootFunctor')
	_add_template_type_printer(obj, prefix, 'MultiplyByFactorFunctor')
	_add_template_type_printer(obj, prefix, 'AddValueFunctor')
	_add_template_type_printer(obj, prefix, 'IncrementFunctor')
	_add_template_type_printer(obj, prefix, 'MaxFunctor')
	_add_template_type_printer(obj, prefix, 'MinFunctor')
	_add_template_type_printer(obj, prefix, 'LowerLimitFunctor')
	_add_template_type_printer(obj, prefix, 'UpperLimitFunctor')
	_add_template_type_printer(obj, prefix, 'AbsFunctor')
	_add_template_type_printer(obj, prefix, 'AbsFunctorInplace')


bolt_printer = None
"""Pretty-printer for bolt."""

def register_printers(obj):
	"""Register bolt pretty-printers with objfile `obj`

	Example::

		import sys
		import gdb

		sys.path.insert(0, '..')

		import boltdbg.printers

		boltdbg.printers.register_printers(gdb.current_objfile())

"""
	global _use_gdb_pp
	global bolt_printer

	if _use_gdb_pp:
		gdb.printing.register_pretty_printer(obj, bolt_printer)
	else:
		if obj == None:
			obj = gdb
		obj.pretty_printers.append(bolt_printer)

	_register_type_printers(obj)


def _register_data_printers():
	"""Creates pretty-printers."""
	global bolt_printer
	prefix = 'bolt::'

	bolt_printer = Printer('bolt')

	# hybrid vector
	bolt_printer.add(SimpleDataPrinter, prefix, 'Vector', membername = 'data')

	# images
	bolt_printer.add(BoltRawImagePrinter, prefix, 'DeviceImage', ptr_name = 'device_ptr_')
	bolt_printer.add(BoltUniqueImagePrinter, prefix, 'HostImage', ptr_name = 'host_ptr_')
	bolt_printer.add(BoltRawImagePrinter, prefix, 'UnifiedImage', ptr_name = 'unified_ptr_')
	bolt_printer.add(BoltTextureImagePrinter, prefix, 'TextureImage', texture_object = 'texture_object_')

	# views
	bolt_printer.add(BoltRawImagePrinter, prefix, 'DeviceImageView', ptr_name = 'device_ptr_')
	bolt_printer.add(BoltRawImagePrinter, prefix, 'DeviceImageConstView', ptr_name = 'device_ptr_')
	bolt_printer.add(BoltRawImagePrinter, prefix, 'HostImageView', ptr_name = 'host_ptr_')
	bolt_printer.add(BoltRawImagePrinter, prefix, 'HostImageConstView', ptr_name = 'host_ptr_')
	bolt_printer.add(BoltRawImagePrinter, prefix, 'UnifiedImageView', ptr_name = 'unified_ptr_')
	bolt_printer.add(BoltRawImagePrinter, prefix, 'UnifiedImageConstView', ptr_name = 'unified_ptr_')
	bolt_printer.add(BoltTextureImagePrinter, prefix, 'TextureImageView', texture_object = 'tex_object_')
	bolt_printer.add(BoltTextureImagePrinter, prefix, 'TextureImageConstView', texture_object = 'tex_object_')

	# procedural views
	bolt_printer.add(BoltMemberPrinter, prefix, 'ConstantImageView', children = ['size', 'element'],
			childdata = {'size': 'size_', 'element': 'element_'})
	bolt_printer.add(BoltMemberPrinter, prefix, 'CheckerBoardImageView', children = ['size', 'tile_size', 'white', 'black'],
			childdata = {'size': 'size_', 'tile_size': 'tile_size_', 'white': 'white_', 'black': 'black_'})
	bolt_printer.add(BoltMemberPrinter, prefix, 'BinaryOperatorImageView', children = ['size', 'view1', 'view2'],
			childdata = {'size': 'size_', 'view1': 'view1_', 'view2': 'view2_'})
	bolt_printer.add(BoltMemberPrinter, prefix, 'LinearCombinationImageView', children = ['size', 'view1', 'view2', 'factor1', 'factor2'],
			childdata = {'size': 'size_', 'view1': 'view1_', 'view2': 'view2_', 'factor1': 'factor1_', 'factor2': 'factor2_'})
	bolt_printer.add(BoltMemberPrinter, prefix, 'MultiplicationImageView', children = ['size', 'view1', 'view2'],
			childdata = {'size': 'size_', 'view1': 'view1_', 'view2': 'view2_'})
	bolt_printer.add(BoltMemberPrinter, prefix, 'DivisionImageView', children = ['size', 'view1', 'view2'],
			childdata = {'size': 'size_', 'view1': 'view1_', 'view2': 'view2_'})
	bolt_printer.add(BoltMemberPrinter, prefix, 'MirrorImageView', children = ['size', 'view', 'flips'],
			childdata = {'size': 'size_', 'view': 'view_', 'flips': 'flips_'})
	bolt_printer.add(BoltMemberPrinter, prefix, 'PaddedImageView', children = ['size', 'view', 'offset', 'fill_value'],
			childdata = {'size': 'size_', 'view': 'view_', 'offset': 'offset_', 'fill_value': 'fill_value_'})
	bolt_printer.add(BoltMemberPrinter, prefix, 'UnaryOperatorImageView', children = ['size', 'view', 'operator'],
			childdata = {'size': 'size_', 'view': 'view_', 'operator': 'unary_operator_'})
	bolt_printer.add(BoltMemberPrinter, prefix, 'MeshGridView', children = ['size', 'dimension', 'start', 'increment'],
			childdata = {'size': 'size_', 'dimension': 'dimension_', 'start': 'start_', 'increment': 'increment_'})

	# functors
	bolt_printer.add(BoltFunctorMemberPrinter, prefix, 'IdentityFunctor')
	bolt_printer.add(BoltFunctorMemberPrinter, prefix, 'SquareFunctor')
	bolt_printer.add(BoltFunctorMemberPrinter, prefix, 'SquareRootFunctor')
	bolt_printer.add(BoltFunctorMemberPrinter, prefix, 'MultiplyByFactorFunctor', children = ['factor'],
			childdata = {'factor': 'factor_'})
	bolt_printer.add(BoltFunctorMemberPrinter, prefix, 'AddValueFunctor', children = ['value'],
			childdata = {'value': 'value_'})
	bolt_printer.add(BoltFunctorMemberPrinter, prefix, 'IncrementFunctor', children = ['value'],
			childdata = {'value': 'value_'})
	bolt_printer.add(BoltFunctorMemberPrinter, prefix, 'MaxFunctor', children = ['limit'],
			childdata = {'limit': 'limit_'})
	bolt_printer.add(BoltFunctorMemberPrinter, prefix, 'MinFunctor', children = ['limit'],
			childdata = {'limit': 'limit_'})
	bolt_printer.add(BoltFunctorMemberPrinter, prefix, 'LowerLimitFunctor', children = ['limit', 'replacement'],
			childdata = {'limit': 'limit_', 'replacement': 'replacement_'})
	bolt_printer.add(BoltFunctorMemberPrinter, prefix, 'UpperLimitFunctor', children = ['limit', 'replacement'],
			childdata = {'limit': 'limit_', 'replacement': 'replacement_'})
	bolt_printer.add(BoltFunctorMemberPrinter, prefix, 'AbsFunctor')
	bolt_printer.add(BoltFunctorMemberPrinter, prefix, 'AbsFunctorInplace')


# create bolt_printer
_register_data_printers()
