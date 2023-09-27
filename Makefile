WEIGHTDIR = saved_models
DATADIR = datasets

all: WEIGHTDIR DATADIR saved_models datasets

WEIGHTDIR:
	mkdir -p $(WEIGHTDIR)

DATADIR:
	mkdir -p $(DATADIR)

saved_models: isic segpc

datasets: isic2018 #segpc2021

isic: $(WEIGHTDIR)/isic2018_unet $(WEIGHTDIR)/isic2018_attunet $(WEIGHTDIR)/isic2018_unetpp $(WEIGHTDIR)/isic2018_multiresunet $(WEIGHTDIR)/isic2018_resunet $(WEIGHTDIR)/isic2018_transunet $(WEIGHTDIR)/isic2018_uctransnet $(WEIGHTDIR)/isic2018_missformer 

segpc: $(WEIGHTDIR)/segpc2021_unet $(WEIGHTDIR)/segpc2021_attunet $(WEIGHTDIR)/segpc2021_unetpp $(WEIGHTDIR)/segpc2021_multiresunet $(WEIGHTDIR)/segpc2021_resunet $(WEIGHTDIR)/segpc2021_transunet $(WEIGHTDIR)/segpc2021_uctransnet $(WEIGHTDIR)/segpc2021_missformer 

isic2018: $(DATADIR)/isic2018/inputs $(DATADIR)/isic2018/outputs $(DATADIR)/isic2018/np
segpc2021: $(DATADIR)/segpc2021/np

$(WEIGHTDIR)/isic2018_unet:
	megadl https://mega.nz/file/pNd0xLIB#LqY-e-hdQhq6_dQZpAw_7MxKclMB5DAFMybL5w99OzM --path $(WEIGHTDIR);
	unzip -d $(WEIGHTDIR) $(WEIGHTDIR)/isic2018_unet.zip;
	rm $(WEIGHTDIR)/isic2018_unet.zip;

$(WEIGHTDIR)/isic2018_attunet:
	megadl https://mega.nz/file/5VsBTKgK#vNu_nvuz-9Lktw6aMOxuguQyim1sVnG4QdkGtVX3pEs --path $(WEIGHTDIR);
	unzip -d $(WEIGHTDIR) $(WEIGHTDIR)/isic2018_attunet.zip;
	rm $(WEIGHTDIR)/isic2018_attunet.zip;

$(WEIGHTDIR)/isic2018_unetpp:
	megadl https://mega.nz/file/NcFQUY5D#1mSGOC4GGTA8arWzcM77yyH9GoApciw0mB4pFp18n0Q --path $(WEIGHTDIR);
	unzip -d $(WEIGHTDIR) $(WEIGHTDIR)/isic2018_unetpp.zip;
	rm $(WEIGHTDIR)/isic2018_unetpp.zip;

$(WEIGHTDIR)/isic2018_multiresunet:
	megadl https://mega.nz/file/tIEVAAba#t-5vLCMwlH6hzAri7DJ8ut-eT2vFN5b6qj6Vc3By6_g --path $(WEIGHTDIR);
	unzip -d $(WEIGHTDIR) $(WEIGHTDIR)/isic2018_multiresunet.zip;
	rm $(WEIGHTDIR)/isic2018_multiresunet.zip;

$(WEIGHTDIR)/isic2018_resunet:
	megadl https://mega.nz/file/NAVHSSJa#FwcYG6bKOdpcEorN_nnjWFEx29toSspSiMzFTqIrVW4 --path $(WEIGHTDIR);
	unzip -d $(WEIGHTDIR) $(WEIGHTDIR)/isic2018_resunet.zip;
	rm $(WEIGHTDIR)/isic2018_resunet.zip;

$(WEIGHTDIR)/isic2018_transunet:
	megadl https://mega.nz/file/UM9jkK6B#7rFd9TiOY6pEGt-gDosFopdV78slgpHbj_wKZ4H39OM --path $(WEIGHTDIR);
	unzip -d $(WEIGHTDIR) $(WEIGHTDIR)/isic2018_transunet.zip;
	rm $(WEIGHTDIR)/isic2018_transunet.zip;

$(WEIGHTDIR)/isic2018_uctransnet:
	megadl https://mega.nz/file/RMNQmKoQ#j8zGEuud33eh-tOIZa1dpkReB8DYKt1De75eeR7wLnM --path $(WEIGHTDIR);
	unzip -d $(WEIGHTDIR) $(WEIGHTDIR)/isic2018_uctransnet.zip;
	rm $(WEIGHTDIR)/isic2018_uctransnet.zip;

$(WEIGHTDIR)/isic2018_missformer:
	megadl https://mega.nz/file/EANRiBoQ#E2LC0ZS7LU5OuEdQJ8dYGihjzqpEEotUqLEnEGZ59wU --path $(WEIGHTDIR);
	unzip -d $(WEIGHTDIR) $(WEIGHTDIR)/isic2018_missformer.zip;
	rm $(WEIGHTDIR)/isic2018_missformer.zip;

$(WEIGHTDIR)/segpc2021_unet:
	megadl https://mega.nz/file/EZEjTYyT#UMsliboXuqrsobGHV_mn4jiBrOf_dMZF7hp2aY0o2hI --path $(WEIGHTDIR);
	unzip -d $(WEIGHTDIR) $(WEIGHTDIR)/segpc2021_unet.zip;
	rm $(WEIGHTDIR)/segpc2021_unet.zip;

$(WEIGHTDIR)/segpc2021_attunet:
	megadl https://mega.nz/file/gRVCXCgT#We3_nPsx_xIBXy6-bsg85rQYYzKHut17Zn5HDnh0Aqw --path $(WEIGHTDIR);
	unzip -d $(WEIGHTDIR) $(WEIGHTDIR)/segpc2021_attunet.zip;
	rm $(WEIGHTDIR)/segpc2021_attunet.zip;

$(WEIGHTDIR)/segpc2021_unetpp:
	megadl https://mega.nz/file/JFVSHLxY#EwPpPZ5N0KDaXhDXxyyuQ_HaD2iNiv5hdqplznrP8Os --path $(WEIGHTDIR);
	unzip -d $(WEIGHTDIR) $(WEIGHTDIR)/segpc2021_unetpp.zip;
	rm $(WEIGHTDIR)/segpc2021_unetpp.zip;

$(WEIGHTDIR)/segpc2021_multiresunet:
	megadl https://mega.nz/file/tUN11R5C#I_JpAT7mYDM1q40ulp8TJxnzHFR4Fh3WX_klep62ywE --path $(WEIGHTDIR);
	unzip -d $(WEIGHTDIR) $(WEIGHTDIR)/segpc2021_multiresunet.zip;
	rm $(WEIGHTDIR)/segpc2021_multiresunet.zip;

$(WEIGHTDIR)/segpc2021_resunet:
	megadl https://mega.nz/file/gQ91WBRB#mzIAeEUze4cAi74dMa3rqivGdYtzpKqDI16vNao7-6A --path $(WEIGHTDIR);
	unzip -d $(WEIGHTDIR) $(WEIGHTDIR)/segpc2021_resunet.zip;
	rm $(WEIGHTDIR)/segpc2021_resunet.zip;

$(WEIGHTDIR)/segpc2021_transunet:
	megadl https://mega.nz/file/5YFBXBoZ#6S8B6MyAsSsr5cNw0-QIIIzF6CgxhEUsOl0xwAknTr8 --path $(WEIGHTDIR);
	unzip -d $(WEIGHTDIR) $(WEIGHTDIR)/segpc2021_transunet.zip;
	rm $(WEIGHTDIR)/segpc2021_transunet.zip;

$(WEIGHTDIR)/segpc2021_uctransnet:
	megadl https://mega.nz/file/hYMShICa#kg5VFhE-m5X0ouE1rc_teaYSb_E15NpbBVE0P_V7WH8 --path $(WEIGHTDIR);
	unzip -d $(WEIGHTDIR) $(WEIGHTDIR)/segpc2021_uctransnet.zip;
	rm $(WEIGHTDIR)/segpc2021_uctransnet.zip;

$(WEIGHTDIR)/segpc2021_missformer:
	megadl https://mega.nz/file/9I1CUJbZ#V6zdx8vZDyPJjHmVgoJH4D86sTuqNu6OuHeUQVB6ees --path $(WEIGHTDIR);
	unzip -d $(WEIGHTDIR) $(WEIGHTDIR)/segpc2021_missformer.zip;
	rm $(WEIGHTDIR)/segpc2021_missformer.zip;

$(DATADIR)/isic2018/inputs:
	curl https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Training_Input.zip -o $(DATADIR)/isic2018/inputs.zip;
	unzip -d $(DATADIR)/isic2018 $(DATADIR)/isic2018/inputs.zip;
	mv $(DATADIR)/isic2018/ISIC2018_Task1-2_Training_Input/ $(DATADIR)/isic2018/inputs/;
	rm $(DATADIR)/isic2018/inputs.zip;

$(DATADIR)/isic2018/outputs:
	curl https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Training_GroundTruth.zip -o $(DATADIR)/isic2018/outputs.zip;
	unzip -d $(DATADIR)/isic2018 $(DATADIR)/isic2018/outputs.zip;
	mv $(DATADIR)/isic2018/ISIC2018_Task1_Training_GroundTruth/ $(DATADIR)/isic2018/outputs/;
	rm $(DATADIR)/isic2018/outputs.zip;

$(DATADIR)/isic2018/np: $(DATADIR)/isic2018/inputs $(DATADIR)/isic2018/outputs
	python $(DATADIR)/prepare_isic2018.py $(DATADIR)

$(DATADIR)/segpc2021/np:
	unzip -d $(DATADIR)/segpc2021/ $(DATADIR)/archive.zip;
	# remove superfluous layers of TCIA_SegPC_dataset/TCIA_SegPC_dataset/TCIA_SegPC_dataset
	mv $(DATADIR)/segpc2021/TCIA_SegPC_dataset $(DATADIR)/segpc2021/temp;
	mv $(DATADIR)/segpc2021/temp/TCIA_SegPC_dataset/TCIA_SegPC_dataset $(DATADIR)/segpc2021;
	rm -rf $(DATADIR)/segpc2021/temp;
	# remove superfluous train/train/train
	mv $(DATADIR)/segpc2021/TCIA_SegPC_dataset/train $(DATADIR)/segpc2021/TCIA_SegPC_dataset/temp;
	mv $(DATADIR)/segpc2021/TCIA_SegPC_dataset/temp/train/train $(DATADIR)/segpc2021/TCIA_SegPC_dataset;
	rm -rf $(DATADIR)/segpc2021/TCIA_SegPC_dataset/temp;
	# remove superfluous validation/validation
	mv $(DATADIR)/segpc2021/TCIA_SegPC_dataset/validation $(DATADIR)/segpc2021/TCIA_SegPC_dataset/temp;
	mv $(DATADIR)/segpc2021/TCIA_SegPC_dataset/temp/validation $(DATADIR)/segpc2021/TCIA_SegPC_dataset;
	rm -rf $(DATADIR)/segpc2021/TCIA_SegPC_dataset/temp;
	python $(DATADIR)/prepare_segpc2021.py $(DATADIR)

clean: rm -rf WEIGHTDIR DATADIR
