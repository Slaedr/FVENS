Coding style guidelines
=======================

+ Please allocate variables as close to the point of use as possible, and deallocate them (if applicable) as soon as possible after use. Declare and allocate variables inside the smallest scope possible [this will help catch bugs caused by the use of unintended variables]. Ideally, containers (like `std::vector`) should be used, in which case the destructor will be invoked automatically. If you write a class, make sure to write a destructor that does all necessary deallocation and cleanup. Deallocation of dynamically allocated memory _must_ be done even if you think it's fine not do so and that the operating system will take care of it on program termination. [This will help minimize memory use when combining parts of the code to write new driver functions.]

+ Please use lots of functions. Long functions doing many different things are to be avoided. [Helps avoid bugs because scopes and lifetimes of variables get reduced; see the point above.] If a function exists solely to break up a larger function, or more generally if it's never needed outside of the file it's defined in, declare it `static`. If such a function is a class member, make it `private`.

+ Avoid long files with lots of barely-related functionality. Pleae add new files for new functionality as and when needed. This keeps files more manageable and compile times bearable.

+ Do not hesitate to use new C++ 14 features where they are useful. In any case, please build all files in C++ 14 mode. But avoid C++ 17, as some compilers still don't support it.

Development guidelines
======================

+ Please make sure your code builds with `-Wall -Werror` in the compiler command line. In the rare case you absolutely must have a warning while building a file and there's no reasonable way to get rid of it, make an exception for only that kind of warning for only that file, using `-Wno-error=<stupid warning>`.

+ Please run the tests regularly and make sure they pass.

+ Please try to write unit tests for any new functionality, and use [CTest](https://cmake.org/cmake/help/latest/manual/ctest.1.html) to add them to the project.
