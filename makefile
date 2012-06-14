all: sudoku

preprocessing.o:
	g++ preprocessing.c -c

basicORC.o: preprocessing.o
	g++ preprocessing.o basicOCR.cpp -c

sudoku: preprocessing.o basicORC.o
	g++ -o sudoku preprocessing.o basicOCR.o sudoku.cpp sudokuHelper.cpp -lopencv_core -lopencv_imgproc -lopencv_calib3d -lopencv_video -lopencv_features2d -lopencv_ml -lopencv_highgui -lopencv_objdetect -lopencv_contrib -lopencv_legacy

clean:
	rm -rf *.o sudoku
