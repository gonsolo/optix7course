SUBDIRS = softoptix example01_helloOptix example02_pipelineAndRayGen example03_inGLFWindow \
          example04_firstTriangleMesh example05_firstSBTData example06_multipleObjects \
          example07_firstRealModel example08_addingTextures example09_shadowRays

.PHONY: all clean

all clean:
	@for dir in $(SUBDIRS); do echo "$@ $$dir"; $(MAKE) -s -C $$dir -f Makefile $@; done

