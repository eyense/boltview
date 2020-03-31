# Sample gdb script for pretty printers inclusion.
# Workflow:
# - modify sys.path.insert() from below such that gdb can find new pretty
#   printers. It accepts also relative paths.
# - add breakpoints following example below
# - run gdb
# - from the gdb environment run:
#   source path/to/printers.gdb

# setup pretty printing data structures on more lines
set print pretty on

# import pretty printers
python
import sys
sys.path.insert(0, "path/to/pretty/printers/module")
import boltviewdbg
boltdbg.register_printers(None)
end

# unrecognized bbreakpoints converted pending (they are automatically loaded
# when shared library is loaded)
set breakpoint pending on

# set your break points
break source.cc:158

# run debugging
run
