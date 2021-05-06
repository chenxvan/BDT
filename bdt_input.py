from matplotlib import pyplot as plt
import uproot
import numpy as np
import pandas as pd

#my_file = uproot.open("/nfs/dust/cms/user/xuan/Zprime_102X/BDT_Out/uhh2.AnalysisModuleRunner.MC.ZprimeToTT_2016v3.root")                      
#my_file_mc = uproot.open("/nfs/dust/cms/user/xuan/Zprime_102X/BDT_Out/uhh2.AnalysisModuleRunner.MC.TT_TuneCUETP8M2T4_2016v3.root")            
my_file_bkg1 = uproot.open("/nfs/dust/cms/user/xuan/Zprime_102X/BDT_Out/uhh2.AnalysisModuleRunner.MC.WJetsToLNu_HT-100To200_2018.root")
my_file_bkg2 = uproot.open("/nfs/dust/cms/user/xuan/Zprime_102X/BDT_Out/uhh2.AnalysisModuleRunner.MC.WJetsToLNu_HT-1200To2500_2018.root")
my_file_bkg3 = uproot.open("/nfs/dust/cms/user/xuan/Zprime_102X/BDT_Out/uhh2.AnalysisModuleRunner.MC.WJetsToLNu_HT-200To400_2018.root")
my_file_bkg4 = uproot.open("/nfs/dust/cms/user/xuan/Zprime_102X/BDT_Out/uhh2.AnalysisModuleRunner.MC.WJetsToLNu_HT-2500ToInf_2018.root")
my_file_bkg5 = uproot.open("/nfs/dust/cms/user/xuan/Zprime_102X/BDT_Out/uhh2.AnalysisModuleRunner.MC.WJetsToLNu_HT-400To600_2018.root")
my_file_bkg6 = uproot.open("/nfs/dust/cms/user/xuan/Zprime_102X/BDT_Out/uhh2.AnalysisModuleRunner.MC.WJetsToLNu_HT-600To800_2018.root")
#my_file_bkg7 = uproot.open("/nfs/dust/cms/user/xuan/Zprime_102X/BDT_Out/uhh2.AnalysisModuleRunner.MC.WJetsToLNu_HT-70To100_2018.root")        
my_file_bkg8 = uproot.open("/nfs/dust/cms/user/xuan/Zprime_102X/BDT_Out/uhh2.AnalysisModuleRunner.MC.WJetsToLNu_HT-800To1200_2018.root")

#my_file_bkg = uproot.open("/nfs/dust/cms/user/xuan/Zprime_102X/BDT_Out/uhh2.AnalysisModuleRunner.MC.WJetsToLNu_2016v3.root")                  

my_file = uproot.open("/nfs/dust/cms/user/xuan/Zprime_102X/BDT_Out/uhh2.AnalysisModuleRunner.MC.ZprimeToTT_2018.root")
my_file_mc = uproot.open("/nfs/dust/cms/user/xuan/Zprime_102X/BDT_Out/uhh2.AnalysisModuleRunner.MC.TTToSemiLeptonic_2018.root")
#my_file_bkg = uproot.open("/nfs/dust/cms/user/xuan/Zprime_102X/BDT_Out/uhh2.AnalysisModuleRunner.MC.WJetsToLNu_2018.root")                    
tree = my_file['AnalysisTree']
tree_mc = my_file_mc['AnalysisTree']
#tree_bkg = my_file_bkg['AnalysisTree']                                                                                                        
#tree_bkg1 = my_file_bkg1['AnalysisTree']                                                                                                      
tree_bkg2 = my_file_bkg2['AnalysisTree']
tree_bkg3 = my_file_bkg3['AnalysisTree']
tree_bkg4 = my_file_bkg4['AnalysisTree']
tree_bkg5 = my_file_bkg5['AnalysisTree']
tree_bkg6 = my_file_bkg6['AnalysisTree']
#tree_bkg7 = my_file_bkg7['AnalysisTree']                                                                                                      
tree_bkg8 = my_file_bkg8['AnalysisTree']
