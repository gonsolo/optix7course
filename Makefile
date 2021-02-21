SUBDIRS = example01_helloOptix example02_pipelineAndRayGen example03_inGLFWindow example04_firstTriangleMesh

all: $(SUBDIRS)
$(SUBDIRS):
	$(MAKE) -C $@

.PHONY: all $(SUBDIRS)
