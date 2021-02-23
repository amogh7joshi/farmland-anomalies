install:
	python3 -m pip install -r requirements.txt
	chmod +x scripts/expand.sh
	chmod +x scripts/preprocess.sh
hide:
	chmod u+x scripts/constant.sh
	scripts/constant.sh hide
show:
	chmod u+x scripts/constant.sh
	scripts/constant.sh show
