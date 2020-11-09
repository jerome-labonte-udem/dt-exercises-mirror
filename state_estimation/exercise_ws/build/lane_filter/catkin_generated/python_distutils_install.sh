#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/code/exercise_ws/src/lane_filter"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/code/exercise_ws/install/lib/python3/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/code/exercise_ws/install/lib/python3/dist-packages:/code/exercise_ws/build/lane_filter/lib/python3/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/code/exercise_ws/build/lane_filter" \
    "/usr/bin/python3" \
    "/code/exercise_ws/src/lane_filter/setup.py" \
     \
    build --build-base "/code/exercise_ws/build/lane_filter" \
    install \
    --root="${DESTDIR-/}" \
    --install-layout=deb --prefix="/code/exercise_ws/install" --install-scripts="/code/exercise_ws/install/bin"
