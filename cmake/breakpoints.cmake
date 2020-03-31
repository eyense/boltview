# Macro breakpoints extract breakpoints from files from FILE_LIST and saves them
# to the OUTPUT file.
#
# Author: Jakub Klener jakub.klener@eyen.eu
#
macro(breakpoints TARGET OUTPUT FILE_LIST)
	add_custom_target(${TARGET} ALL DEPENDS ${OUTPUT})
	set(templist "")
	foreach(f ${${FILE_LIST}})
		LIST(APPEND templist "${CMAKE_CURRENT_SOURCE_DIR}/${f}")
	ENDFOREACH(f)
	add_custom_command(
		OUTPUT ${OUTPUT}
		COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/cmake/breakpoints.sh "${OUTPUT}" ${templist}
		DEPENDS ${${FILE_LIST}}
	)
endmacro(breakpoints)
