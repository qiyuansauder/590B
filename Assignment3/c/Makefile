## Below are the Make file

CC := gcc
CFLAGS := -Wall -g  -lgsl -lgslcblas -lm

TARGET := full

OBJS := full.o lib/randist_mv.o lib/panel.o lib/multi_regress.o

$(TARGET) : $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS)

lib/%.o : lib/%.c
	$(CC) $(CFLAGS) -o $@ -c $<

$(OBJS): lib/wu_randist.h lib/wu_bayes.h

