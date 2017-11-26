TODO: this directory is needed because cython always tries to detect context
package, i.e., always compiles and installs to $PROJROOT/graphemb.* (or
$PROJROOT/graphemb/server.* etc.). As a result, we have to create a
non-package directory. Any better way to do this?
