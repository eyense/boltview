# Macro breakpoints extract breakpoints from files from FILE_LIST and saves them
# to the OUTPUT file.
#
# Author: Jakub Klener jakub.klener@eyen.eu
#
macro(breakpoints TARGET_NAME OUTPUT_FILE FILE_LIST)
	add_custom_target(${TARGET_NAME} ALL DEPENDS ${OUTPUT_FILE})
	set(templist "")
	foreach(f ${${FILE_LIST}})
		LIST(APPEND templist "${f}")
	ENDFOREACH(f)
	add_custom_command(
		OUTPUT ${OUTPUT_FILE}
		COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/cmake/breakpoints.sh ${OUTPUT_FILE} ${templist}
		DEPENDS ${${FILE_LIST}}
	)
endmacro(breakpoints)
