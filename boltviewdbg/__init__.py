"""Package with ecip pretty printers"""


# python2 compatibility
from __future__ import absolute_import

__copyright__ = 'Copyright 2017 Eyen SE'
__author__ =  'Jakub Klener jakub.klener@eyen.eu'

import gdb

def register_printers(obj):
	"""Register ecip pretty-printers with objfile `obj`.

	The proxy method for .printers.register_printers, see `printers.py`.

	Example::

		import sys
		import gdb

		sys.path.insert(0, "..")

		import boltviewdbg

		boltviewdbg.register_printers(gdb.current_objfile())

	"""
	# Load the pretty-printers.
	from .printers import register_printers
	register_printers(obj)
