# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = "/Applications/CMake 2.8-11.app/Contents/bin/cmake"

# The command to remove a file.
RM = "/Applications/CMake 2.8-11.app/Contents/bin/cmake" -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = "/Applications/CMake 2.8-11.app/Contents/bin/ccmake"

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/jimeiyang/Projects/ObjectSegmentation/densecrf

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/jimeiyang/Projects/ObjectSegmentation/densecrf/build

# Include any dependencies generated for this target.
include examples/CMakeFiles/test_permutohedral.dir/depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/test_permutohedral.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/test_permutohedral.dir/flags.make

examples/CMakeFiles/test_permutohedral.dir/test_permutohedral.cpp.o: examples/CMakeFiles/test_permutohedral.dir/flags.make
examples/CMakeFiles/test_permutohedral.dir/test_permutohedral.cpp.o: ../examples/test_permutohedral.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /Users/jimeiyang/Projects/ObjectSegmentation/densecrf/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object examples/CMakeFiles/test_permutohedral.dir/test_permutohedral.cpp.o"
	cd /Users/jimeiyang/Projects/ObjectSegmentation/densecrf/build/examples && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/test_permutohedral.dir/test_permutohedral.cpp.o -c /Users/jimeiyang/Projects/ObjectSegmentation/densecrf/examples/test_permutohedral.cpp

examples/CMakeFiles/test_permutohedral.dir/test_permutohedral.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_permutohedral.dir/test_permutohedral.cpp.i"
	cd /Users/jimeiyang/Projects/ObjectSegmentation/densecrf/build/examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /Users/jimeiyang/Projects/ObjectSegmentation/densecrf/examples/test_permutohedral.cpp > CMakeFiles/test_permutohedral.dir/test_permutohedral.cpp.i

examples/CMakeFiles/test_permutohedral.dir/test_permutohedral.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_permutohedral.dir/test_permutohedral.cpp.s"
	cd /Users/jimeiyang/Projects/ObjectSegmentation/densecrf/build/examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /Users/jimeiyang/Projects/ObjectSegmentation/densecrf/examples/test_permutohedral.cpp -o CMakeFiles/test_permutohedral.dir/test_permutohedral.cpp.s

examples/CMakeFiles/test_permutohedral.dir/test_permutohedral.cpp.o.requires:
.PHONY : examples/CMakeFiles/test_permutohedral.dir/test_permutohedral.cpp.o.requires

examples/CMakeFiles/test_permutohedral.dir/test_permutohedral.cpp.o.provides: examples/CMakeFiles/test_permutohedral.dir/test_permutohedral.cpp.o.requires
	$(MAKE) -f examples/CMakeFiles/test_permutohedral.dir/build.make examples/CMakeFiles/test_permutohedral.dir/test_permutohedral.cpp.o.provides.build
.PHONY : examples/CMakeFiles/test_permutohedral.dir/test_permutohedral.cpp.o.provides

examples/CMakeFiles/test_permutohedral.dir/test_permutohedral.cpp.o.provides.build: examples/CMakeFiles/test_permutohedral.dir/test_permutohedral.cpp.o

# Object files for target test_permutohedral
test_permutohedral_OBJECTS = \
"CMakeFiles/test_permutohedral.dir/test_permutohedral.cpp.o"

# External object files for target test_permutohedral
test_permutohedral_EXTERNAL_OBJECTS =

examples/test_permutohedral: examples/CMakeFiles/test_permutohedral.dir/test_permutohedral.cpp.o
examples/test_permutohedral: examples/CMakeFiles/test_permutohedral.dir/build.make
examples/test_permutohedral: src/libdensecrf.a
examples/test_permutohedral: src/liboptimization.a
examples/test_permutohedral: external/liblbfgs.a
examples/test_permutohedral: examples/CMakeFiles/test_permutohedral.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable test_permutohedral"
	cd /Users/jimeiyang/Projects/ObjectSegmentation/densecrf/build/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_permutohedral.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/CMakeFiles/test_permutohedral.dir/build: examples/test_permutohedral
.PHONY : examples/CMakeFiles/test_permutohedral.dir/build

examples/CMakeFiles/test_permutohedral.dir/requires: examples/CMakeFiles/test_permutohedral.dir/test_permutohedral.cpp.o.requires
.PHONY : examples/CMakeFiles/test_permutohedral.dir/requires

examples/CMakeFiles/test_permutohedral.dir/clean:
	cd /Users/jimeiyang/Projects/ObjectSegmentation/densecrf/build/examples && $(CMAKE_COMMAND) -P CMakeFiles/test_permutohedral.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/test_permutohedral.dir/clean

examples/CMakeFiles/test_permutohedral.dir/depend:
	cd /Users/jimeiyang/Projects/ObjectSegmentation/densecrf/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/jimeiyang/Projects/ObjectSegmentation/densecrf /Users/jimeiyang/Projects/ObjectSegmentation/densecrf/examples /Users/jimeiyang/Projects/ObjectSegmentation/densecrf/build /Users/jimeiyang/Projects/ObjectSegmentation/densecrf/build/examples /Users/jimeiyang/Projects/ObjectSegmentation/densecrf/build/examples/CMakeFiles/test_permutohedral.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/CMakeFiles/test_permutohedral.dir/depend

