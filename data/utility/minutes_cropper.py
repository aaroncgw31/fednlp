import nltk
from nltk.corpus import CategorizedPlaintextCorpusReader

def crop_text(raw_text,start_dict,end_dict):

    sloc=[]
    eloc=[]
    
    for item in start_dict.values():
        slst = re.search(item,raw_text)
        
        if slst != None:
            sloc.extend(slst.span())
    
    for item in end_dict.values():
        elst = re.search(item,raw_text)
        
        if elst != None:
            eloc.extend(elst.span())
    
    if len(sloc) != 0:
        return raw_text[min(sloc):max(eloc)]
    else:
        print(raw_text)
        return None

def saveFile(fname,year,text):
    main_directory = file_path_prefix + 'minutes_cropped/'
    os.chdir(main_directory)
    directory = main_directory + str(year) + '/'
    
    if not os.path.exists(directory):#check if directory exists
        os.makedirs(directory)
        
    if not os.path.isfile(fname):#check if file name already exists
        os.chdir(directory)
        file= open(fname, 'w')
        file.write(text)
        file.close()
                
doc_start= {}
doc_start[0] = "Staff Review of the Economic Situation"
doc_start[1] = re.compile('The information (reviewed|received|provided)')
doc_start[2] = "The Committee then turned to a discussion of the economic outlook"
doc_start[3] = re.compile('The information  (reviewed|received|provided)')



doc_end ={}
doc_end[0] = re.compile('(At the conclusion of) (this|the) (discussion|meeting)')
doc_end[1] = re.compile('(?i)The Committee voted to authorize')
doc_end[2] = re.compile('(?i)The vote encompassed approval of')


if __main__ == '__name__':
    corpus_root = './minutes'
    data_m = CategorizedPlaintextCorpusReader(corpus_root, r'.*\.txt', cat_pattern=r'(\w+)/*')
    data_fileids = data_m.fileids()
    
    for f in data_fileids:
        year,fname = f.split('/')
        cropped_text = crop_text(data_m.raw(f),doc_start,doc_end)
        saveFile(fname,year,cropped_text)
