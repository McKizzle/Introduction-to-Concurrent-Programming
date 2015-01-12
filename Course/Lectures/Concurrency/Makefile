FILE = Concurrency_Introduction

#Themes: CambridgeUS
THEME = Boadilla

beamer:
	pandoc -t beamer -s $(FILE).md -o $(FILE)-$@.pdf  --slide-level=2 -V theme:$(THEME)

notes: 
	pandoc -t beamer -s $(FILE).md -o $(FILE)-$@.pdf  --slide-level=2 -V classoption=notes=only

latex:
	pandoc -t beamer -s $(FILE).md -o $(FILE)-$@.tex  --slide-level=2 -V theme:$(THEME)
	pandoc -t beamer -s $(FILE).md -o $(FILE)-$@.tex  --slide-level=2 -V classoption=notes=only -V theme:$(THEME)

all: beamer notes

clean:
	rm -f *.pdf *.html

