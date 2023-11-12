find_path(LIBUSB_INCLUDE_DIR
  NAMES
  libusb.h
  PATH_SUFFIXES
  "include"
  "libusb"
  "libusb-1.0"
)

find_library(LIBUSB_LIBRARY
  NAMES
  usb-1.0
  PATH_SUFFIXES
  "lib"
  "lib64"
)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set LIBUSB_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(libusb DEFAULT_MSG LIBUSB_LIBRARY LIBUSB_INCLUDE_DIR)

mark_as_advanced(LIBUSB_INCLUDE_DIR LIBUSB_LIBRARY)

set(LIBUSB_INCLUDE_DIRS ${LIBUSB_INCLUDE_DIR})
set(LIBUSB_LIBRARIES ${LIBUSB_LIBRARY})