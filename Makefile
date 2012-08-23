

# install prefix (use as command line variable)
PREFIX ?= $(pwd)

# install prefix (use as environment variable)
FISHLIB_INSTALL ?= $(PREFIX)

.PHONY : clib install

default : clib

all : clib pyfish

clib : 
	@make -C src

pyfish : clib
	@make -C pyfish

install : clib
	mkdir -p $(FISHLIB_INSTALL)/include; cp include/* $(FISHLIB_INSTALL)/include
	mkdir -p $(FISHLIB_INSTALL)/bin; cp bin/* $(FISHLIB_INSTALL)/bin
	mkdir -p $(FISHLIB_INSTALL)/lib; cp lib/* $(FISHLIB_INSTALL)/lib

clean :
	@make -C src clean
	@make -C pyfish clean
	@rm -rf lib bin include
