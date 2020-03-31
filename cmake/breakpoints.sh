#!/bin/sh
# extract all breakpoints to one file from line annotated by // DEBUG_BREAK
# The breakpoints can be loaded to gdb by: source breakpoints.gdb
# If you are using break points for shared library, be sure that pending break
# points are set to be automatically loaded before you try to load the
# breakpoints:
# set breakpoint pending on
#
# Author: Jakub Klener jakub.klener@eyen.eu
#

set -e

#printf "Generating breakpoints into: %s\\n" "$1"

output=$1
shift

# clear file
>"$output"

while [ $# -gt 0 ]
do
	#printf -- "-- processing %s...\\n" "$1"
	grep -Hn "// DEBUG_BREAK" $1 | cut -d":" -f1-2 | sed 's:^.*/:break :' >> "$output"
	shift
done
