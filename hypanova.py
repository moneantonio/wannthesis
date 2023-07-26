import csv
import os
import json
import argparse
import time
import numpy as np , numpy.random
import pandas as pd
import random
import typing
import logging
from fanova import fANOVA
import fanova.visualizer
from ConfigSpace.read_and_write import json as cs_json
import pickle
import statsmodels.api as sm
from statsmodels.formula.api import ols,poisson
from statsmodels.stats.anova import anova_lm
from bioinfokit.analys import stat
from patsy import ModelDesc,dmatrix,demo_data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.factorplots import interaction_plot
import statsmodels.stats.multicomp as mc
from ConfigSpace import ConfigurationSpace, Float
import warnings
warnings.filterwarnings("ignore")
#credits to https://github.com/janvanrijn/openml-pimp/blob/349043fb324a29e2ee5463d1c85bfd40213fe4fe/examples/experiments/run_pimp_on_arff.py#L123 
#credits to https://github.com/janvanrijn/openml-pimp/blob/349043fb324a29e2ee5463d1c85bfd40213fe4fe/examples/plot/plot_fanova_aggregates.py#L131
#credits to https://automl.github.io/fanova/manual.html
#credits to https://ada.liacs.leidenuniv.nl/papers/ShaEtAl19.pdf
#credits to https://ada.liacs.nl/papers/RijHut18.pdf



def createConfigs():
    howmany = 30
    json_path = '/Users/antoniomone/Desktop/Uni/LEIDEN/THESIS/WANN/brain-tokyo-workshop/WANNRelease/prettyNeatWann/p/atariJsons2/'
    cs = ConfigurationSpace(seed=42,space={
            "task":"atari_stack128_ram",
            "alg_nMean": 6,
            "alg_nReps": 1,
            "maxGen": 2048,
            "popSize": 32,
            "prob_initEnable": (0.0,0.5),
            "prob_mutAct":  (0.05,0.55),
            "prob_addNode": (0.05,0.55),
            "prob_addConn": (0.05,0.55),
            "prob_enable":  (0.05,0.55),
            "save_mod": 1,
            "ram": 0
    })
    samp = cs.sample_configuration(size=howmany)
    for i in range(howmany):
        samp_dic = samp[i].get_dictionary()
        with open(json_path+'configspace'+str(i+1)+'.json', 'w') as f:
            json.dump(samp_dic, f)

def importAndCreate(argv):
    #path_to_csvs = "/Users/antoniomone/Desktop/Uni/LEIDEN/THESIS/WANN/brain-tokyo-workshop/WANNRelease/prettyNeatWann/hypCSVs/"
    path_to_csvs = "/Users/antoniomone/Desktop/Uni/LEIDEN/THESIS/WANN/brain-tokyo-workshop/WANNRelease/prettyNeatWannCuPy_colab/hypsRes/"
    game = argv.game
    path_to_configs = '/Users/antoniomone/Desktop/Uni/LEIDEN/THESIS/WANN/brain-tokyo-workshop/WANNRelease/prettyNeatWannCuPy_colab/p/atariJsons2/'
    folder = sorted(os.scandir(path_to_csvs), key=lambda e: e.name)
    values = []
    scores_df = pd.DataFrame(columns=['score'])
    for elem in folder :
        if elem.is_file():
            name = elem.name
            if name.startswith(game):
                print(name)
                df = pd.read_csv(path_to_csvs+name)
                avg_score = df.iloc[-1].tolist()[1]
                scores_df.loc[len(scores_df)] = avg_score
                #values.append(df.iloc[-1].tolist())
    
    print(scores_df)
    scores_df.to_csv(path_to_csvs+'_'+game+'/'+game+"_scores.csv")
    folder = sorted(os.scandir(path_to_configs),key=lambda e:e.name)
    hyps_df = pd.DataFrame(columns=['alg_nMean', 'alg_nReps', 'maxGen', 'popSize', 'prob_addConn',
                                    'prob_addNode', 'prob_enable', 'prob_initEnable', 'prob_mutAct',
                                    'ram', 'save_mod', 'task'])
    for elem in folder:
        if elem.is_file():
            name = elem.name
            if name.startswith("config"): #.DS_Store file
                #checker = name.replace(".json","")
                print(name)
                data = json.load(open(path_to_configs+name))
                #print(data)
                hyps_df.loc[len(hyps_df)] = data

    print(hyps_df)
    hyps_df = hyps_df.drop(['alg_nMean', 'alg_nReps', 'maxGen', 'popSize','ram', 'save_mod','task'],axis=1)
    print(hyps_df)
    hyps_df.to_csv(path_to_csvs+'_'+game+'/'+game+"_hyps.csv")
    
def calculate_cutoff_value(medians: pd.DataFrame, column_name: str, n_combi_params: typing.Optional[int]):
    medians_sorted = medians[medians['n_hyperparameters'] > 1].sort_values(column_name)
    cutoff = 0.0
    if n_combi_params is not None and len(medians_sorted) > n_combi_params:
        cutoff = medians_sorted[column_name][-1 * n_combi_params]
    return cutoff    
    
def boxplots_variance_contrib(df: pd.DataFrame, output_file: str, n_combi_params: int, log_scale: bool):
    elements = ['n_hyperparameters', 'importance_variance', 'importance_max_min']
    medians = df.groupby('hyperparameter')[elements].median()
    df = df.join(medians, on='hyperparameter', how='left', rsuffix='_median')

    # vanilla boxplots
    plt.clf()
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    cutoff_value_variance = calculate_cutoff_value(medians, 'importance_variance', n_combi_params) - 0.000001 if n_combi_params > 0 else 1.0
    sns.boxplot(x='hyperparameter', y='importance_variance',
                data=df.query(
                    'n_hyperparameters == 1 or importance_variance_median >= %f' % cutoff_value_variance).sort_values(
                    'importance_variance_median'), ax=ax1)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.set_ylabel('Variance Contribution')
    ax1.set_xlabel(None)
    if log_scale:
        ax1.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_file)
    logging.info('stored figure to %s' % output_file)    
      
def packagetest(argv):
    
    #df = calc_man()
    #folder = '/Users/antoniomone/Desktop/Uni/LEIDEN/THESIS/WANN/brain-tokyo-workshop/WANNRelease/prettyNeatWann/hypCSVs/_'
    folder = '/Users/antoniomone/Desktop/Uni/LEIDEN/THESIS/WANN/brain-tokyo-workshop/WANNRelease/prettyNeatWannCuPy_colab/hypsRes/_'
    tot_folder = '/Users/antoniomone/Desktop/Uni/LEIDEN/THESIS/WANN/brain-tokyo-workshop/WANNRelease/prettyNeatWannCuPy_colab/hypsRes/'
    tot_folder = os.scandir(tot_folder)
    output_folder = '/Users/antoniomone/Desktop/Uni/LEIDEN/THESIS/WANN/brain-tokyo-workshop/WANNRelease/prettyNeatWannCuPy_colab/hypsRes/output/'
    game = argv.game
    print(f"------------------------------GAME-->{game}-------------------------------------")
    path = folder+game+'/'
    file_hyps = game+'_hyps.csv'
    file_scores = game+'_scores.csv'
    X = np.loadtxt(path + file_hyps, delimiter=",",skiprows=1)
    X_DF = pd.DataFrame(columns=['prob_addConn','prob_addNode', 'prob_enable', 'prob_initEnable', 'prob_mutAct'])
    for i in range(len(X)):
        X_DF.loc[len(X_DF)] = np.asarray(X[i][1:])
    
    
    Y = np.loadtxt(path + file_scores, delimiter=",",skiprows=1)
    '''cs = ConfigurationSpace(seed=42,
                            space={
                                "prob_initEnable": (0.0,0.55),
                                "prob_mutAct":  (0.05,0.55),
                                "prob_addNode": (0.05,0.55),
                                "prob_addConn": (0.05,0.55),
                                "prob_enable":  (0.05,0.55),
                                }
                            )'''
    cs = ConfigurationSpace(seed=42,
                            space={
                                "Initially enabled": (0.0,0.55),
                                "Change activation":  (0.05,0.55),
                                "New node": (0.05,0.55),
                                "New connection": (0.05,0.55),
                                "Enable connection":  (0.05,0.55),
                                }
                            )
    #print(cs.get_hyperparameter_names())
    #f = fANOVA(NEWX,Y,config_space=cs)
    X_DF = X_DF.rename(columns={"prob_initEnable":"Initially enabled","prob_mutAct":"Change activation",
                                "prob_addNode":"New node","prob_addConn":"New connection","prob_enable":"Enable connection"}
                      )
    f = fANOVA(X_DF,Y,config_space=cs)
    #pairw = f.quantify_importance(("prob_initEnable","prob_mutAct","prob_addNode","prob_addConn","prob_enable"))
    pairw = f.quantify_importance(("Initially enabled","Change activation","New node","New connection","Enable connection"))
    print("---------------------------------------------------------------------------------")
    print("-----------------------------------FANOVA TEST-----------------------------------")
    print("---------------------------------------------------------------------------------")
    som = 0
    for k in pairw:
        print(k,pairw[k])
        som += pairw[k]['individual importance']
    print(som)
    vis = fanova.visualizer.Visualizer(f, cs, path+'plots/')
    '''folder = sorted(os.scandir(path+'plots/interactive_plots/'), key=lambda e: e.name)
    print("-------------------------------SHOWING COMB MARGINAL-----------------------------")
    for elem in folder:
        if elem.is_file():
            file = elem.name
            figx = pickle.load(open(path+'plots/interactive_plots/'+file, 'rb'))
            figx.show()'''
    if argv.visu == 0:
        print("--------------------------------------VISUAL-------------------------------------")
        for i in range(5):
            print(f"--------------------------------SHOWING H{i} MARGINAL------------------------------")
            vis.generate_marginal(i)
            vis.plot_marginal(i)    
    result = list()      
    if argv.visu == 1:  
        for elem in tot_folder:
            if elem.is_dir():
                if elem.name.startswith("_"):
                    newfol = os.scandir(elem.path)
                    print(elem.name)
                    for file in newfol:
                        if file.is_file():
                            if file.name.endswith("csv"):
                                #print(file)
                                if file.name.endswith("hyps.csv"):
                                    t_X = np.loadtxt(path + file_hyps, delimiter=",",skiprows=1)
                                    #temp_X = pd.DataFrame(columns=['prob_addConn','prob_addNode', 'prob_enable', 'prob_initEnable', 'prob_mutAct'])
                                    temp_X = pd.DataFrame(columns=["New connection","New node","Enable connection","Initially enabled","Change activation"])
                                    for i in range(len(t_X)):
                                        temp_X.loc[len(temp_X)] = np.asarray(X[i][1:])
                                if file.name.endswith("scores.csv"):
                                    temp_Y = np.loadtxt(path + file_scores, delimiter=",",skiprows=1)
                    f = fANOVA(temp_X,temp_Y,config_space=cs)
                    vis = fanova.visualizer.Visualizer(f,
                                            cs,
                                            output_folder)
                    for i in range(len(temp_X.columns)):
                                    print("quantifying importance")
                                    importance = f.quantify_importance((temp_X.columns[i],))[(temp_X.columns[i],)]
                                    #print(temp_X.columns[i],importance)
                                    visualizer_res = vis.generate_marginal(temp_X.columns[i], 100)
                                    avg_marginal = np.array(visualizer_res[0])
                                    difference_max_min = max(avg_marginal.reshape((-1,))) - min(avg_marginal.reshape((-1,)))
                                    
                                    current = {
                                        'task_id': elem.name,
                                        'hyperparameter': temp_X.columns[i],#' / '.join(temp_X.columns),
                                        'n_hyperparameters': 1,#len(temp_X.columns),
                                        'importance_variance': importance['individual importance'],
                                        'importance_max_min': difference_max_min,
                                    }
                                    result.append(current)
                                    
        df_result = pd.DataFrame(result)
        df_result.to_csv(output_folder+'fanova_res.csv')
        
        boxplots_variance_contrib(df_result,output_folder+"outputcombi4",n_combi_params=4,log_scale=False)
        print("boxplot created")

def newmain(argv):
    #createConfigs()
    #importAndCreate(argv)
    packagetest(argv)
    #createDfForBoxplots()

        
if __name__ == "__main__":
    ''' Parse input and launch '''
      
    parser = argparse.ArgumentParser(description=('Test HyperTuning'))
    parser.add_argument('-m', '--melted', type=int,\
    help='0 for melted, 1 for extended', default=0)
    parser.add_argument('-v','--visu',type=int,\
    help='visualize plots, 0 yes 1 no', default=1)
    parser.add_argument('-r','--reg',type=int,\
    help='ols or poisson, 0 ols 1 poisson', default=0)
    parser.add_argument('-g','--game',type=str,\
    help='which game?', default="battlezone")
    args = parser.parse_args()
    #calculate()
    newmain(args)








#OLD
    
def main1():
    #analysis()
    df = calc_man()
    print(df)
    formula_gen = 'score ~ h1 +h2 +h3 +h4 + h1:h2 + h1:h3 +h1:h4 +h2:h3 +h2:h4 +h3:h4 +h1:h2:h3 +h1:h2:h4 + h1:h3:h4 +h2:h3:h4 + h1:h2:h3:h4'
    formula = 'score ~ h1 +h2 +h3 +h4 + h1:h2 + h1:h3 +h1:h4 +h2:h3 +h2:h4 +h3:h4'
    model = ols(formula_gen, data=df).fit()
    print(model.summary())
    anova_table = sm.stats.anova_lm(model, typ=3)
    print(anova_table)
    res = stat()
    res.tukey_hsd(df=df, res_var='score', xfac_var='h1', anova_model=formula_gen)
    print(res.tukey_summary)
    print("MINIMUM P from TUKEY",min(res.tukey_summary['p-value']))
    w, pvalue = stats.shapiro(res.anova_model_out.resid)
    print(f"SHAPIRO: W:{w}, P-VALUE {pvalue}")
    '''res.levene(df=df, res_var='value', xfac_var=['h1','h2','h3'])
    print(res.levene_summary)'''
    
def visu(res):    
    sm.qqplot(res.anova_std_residuals, line='45')
    plt.title("Anova Standard Residuals Distribution")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Standardized Residuals")
    plt.show()

    # histogram
    plt.hist(res.anova_model_out.resid, bins='auto', histtype='bar', ec='k')
    plt.title("Residual Plot")
    plt.xlabel("Residuals")
    plt.ylabel('Frequency')
    plt.show()

def main2():
    df = calc_man()
    #print(df)
    df_melt = pd.melt(df, id_vars=df.index, value_vars=['1', '2', '3', '4'])
    #df_melt.columns = ['score', 'h1', 'h2', 'h3','h4']
    print(df_melt.head())
    formula_gen = 'score ~ h1 +h2 +h3 +h4 + h1:h2 + h1:h3 +h1:h4 +h2:h3 +h2:h4 +h3:h4 +h1:h2:h3 +h1:h2:h4 + h1:h3:h4 +h2:h3:h4 + h1:h2:h3:h4'
    formula = 'score ~ h1 +h2 +h3 +h4 + h1:h2 + h1:h3 +h1:h4 +h2:h3 +h2:h4 +h3:h4'
    basic_formula = 'score ~ h1:h2:h3:h4'
    model = ols(formula_gen, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)
    res = stat()
    res.anova_stat(df=df, res_var='score', anova_model=formula_gen)
    print(res.anova_summary)
    # note: if the data is balanced (equal sample size for each group), Type 1, 2, and 3 sums of squares
    # (typ parameter) will produce similar results.
    res = stat()
    res.tukey_hsd(df=df, res_var='score', xfac_var='h1', anova_model=formula_gen)
    print(res.tukey_summary)
    #visu(res)
    w, pvalue = stats.shapiro(model.resid)
    print(f"SHAPIRO: W:{w}, P-VALUE {pvalue}")
    w, pvalue = stats.bartlett(df['h1'], df['h2'], df['h3'], df['h4'])
    print(f"BARTLETT: W:{w}, P-VALUE {pvalue}")
    
def two_factor_version(argv):
    df = calc_man()
    #print(sum(df['score']/len(df))) #1415
    d_melt = pd.melt(df, id_vars=['score'], value_vars=['score','h1','h2', 'h3', 'h4'])
    d_melt.columns = ['score','hyper','prob']
    #print(d_melt.head())
    #print(d_melt.tail())
    if argv.visu == 0:
        plt.figure(figsize=(15,6))
        sns.boxplot(x="prob", y="score", hue="hyper", data=d_melt, palette="Set3");plt.show()
    formula ='score ~ C(hyper)+C(prob)+C(hyper):C(prob)'
    new_formula = 'score ~ C(hyper,Sum) + C(prob,Sum) + C(hyper,Sum)*C(prob,Sum)'
    extra_formula = 'score ~ C(hyper) + C(prob) + C(hyper):C(prob)'
    extra_formula = 'score ~ hyper*prob'
    desc = ModelDesc.from_formula(extra_formula)
    return d_melt,desc

def four_factor_version():
    df = calc_man()
    d_melt = pd.melt(df, id_vars=['h1', 'h2', 'h3', 'h4'], value_vars=['score'])
    d_melt.columns = ['h1', 'h2', 'h3', 'h4','const','score']
    d_melt = d_melt.drop(columns=['const'])
    print(d_melt.head())
    #formula = 'score ~ h1+h2+h3+h4 + h1:h2:h3:h4'
    #formula = 'score ~ h1 +h2 +h3 +h4 + h1:h2 + h1:h3 +h1:h4 +h2:h3 +h2:h4 +h3:h4 +h1:h2:h3 +h1:h2:h4 + h1:h3:h4 +h2:h3:h4 + h1:h2:h3:h4'
    extra_formula = 'score ~ h1*h2*h3*h4'
    desc = ModelDesc.from_formula(extra_formula)
    return d_melt,desc

def meltingtest():
    df = calc_man()
    d_melt = pd.melt(df, id_vars=['index','score'], value_vars=['h1', 'h2', 'h3', 'h4'])
    d_melt.columns = ['index','score','hyper','prob']
    df_wide=pd.pivot_table(data = d_melt, index='hyper', columns = 'prob', values = 'score') #Reshape from long to wide
    print(df_wide.head(5))
    fvalue, pvalue = stats.f_oneway(df_wide[0.10], df_wide[0.15],df_wide[0.20],\
        df_wide[0.25],df_wide[0.30],df_wide[0.40],df_wide[0.55])
    print(fvalue, pvalue)

def pairwise(res,dataf,formula):
    print("----------------------------------------------------------------------------------------")
    print("---------------------------------PAIRWISE COMPARISONS-----------------------------------")
    print("----------------------------------------------------------------------------------------")
    print("perform multiple pairwise comparison (Tukey HSD) for main effect h1")
    res.tukey_hsd(df=dataf, res_var='score', xfac_var='h1', anova_model=formula)
    print(res.tukey_summary)
    print("MINIMUM P from TUKEY",min(res.tukey_summary['p-value']),'for h1')
    counter = 0
    for i in res.tukey_summary['p-value']:
        if i<=0.05:
            counter+=1
    print("h1 pairwise comparisons where p-value<=0.05:",counter)
    print("-----------------------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------------------")
    print("perform multiple pairwise comparison (Tukey HSD) for main effect h2")
    res.tukey_hsd(df=dataf, res_var='score', xfac_var='h2', anova_model=formula)
    print(res.tukey_summary)
    counter = 0
    for i in res.tukey_summary['p-value']:
        if i<=0.05:
            counter+=1
    print("h2 pairwise comparisons where p-value<=0.05:",counter)
    print("MINIMUM P from TUKEY",min(res.tukey_summary['p-value']),'for h2')
    print("-----------------------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------------------")
    print("perform multiple pairwise comparison (Tukey HSD) for main effect h3")
    res.tukey_hsd(df=dataf, res_var='score', xfac_var='h3', anova_model=formula)
    print(res.tukey_summary)
    counter = 0
    for i in res.tukey_summary['p-value']:
        if i<=0.05:
            counter+=1
    print("h3 pairwise comparisons where p-value<=0.05:",counter)
    print("MINIMUM P from TUKEY",min(res.tukey_summary['p-value']),'for h3')
    print("-----------------------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------------------")
    print("perform multiple pairwise comparison (Tukey HSD) for main effect h4")
    res.tukey_hsd(df=dataf, res_var='score', xfac_var='h4', anova_model=formula)
    print(res.tukey_summary)
    counter = 0
    for i in res.tukey_summary['p-value']:
        if i<=0.05:
            counter+=1
    print("h4 pairwise comparisons where p-value<=0.05:",counter)
    print("MINIMUM P from TUKEY",min(res.tukey_summary['p-value']),'for h4')
    print("-----------------------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------------------")

def pairwise2(argv,dataf):
    print("----------------------------------------------------------------------------------------")
    print("---------------------------------PAIRWISE COMPARISONS-----------------------------------")
    print("-----------------------------------------H1---------------------------------------------")
    interaction_groups = "Hyper_" + dataf.h1.astype(str)
    comp = mc.MultiComparison(dataf["score"], interaction_groups)
    post_hoc_res = comp.tukeyhsd()
    if argv.visu==0:
        post_hoc_res.plot_simultaneous(ylabel= "MultiComparison", xlabel= "Score")
        plt.show()
    print(post_hoc_res.summary())
    print("----------------------------------------------------------------------------------------")
    #print("---------------------------------PAIRWISE COMPARISONS-----------------------------------")
    print("-----------------------------------------H2---------------------------------------------")
    interaction_groups = "Hyper_" + dataf.h2.astype(str)
    comp = mc.MultiComparison(dataf["score"], interaction_groups)
    post_hoc_res = comp.tukeyhsd()
    if argv.visu==0:
        post_hoc_res.plot_simultaneous(ylabel= "MultiComparison", xlabel= "Score")
        plt.show()
    print(post_hoc_res.summary())
    print("----------------------------------------------------------------------------------------")
    #print("---------------------------------PAIRWISE COMPARISONS-----------------------------------")
    print("-----------------------------------------H3---------------------------------------------")
    interaction_groups = "Hyper_" + dataf.h3.astype(str)
    comp = mc.MultiComparison(dataf["score"], interaction_groups)
    post_hoc_res = comp.tukeyhsd()
    if argv.visu==0:
        post_hoc_res.plot_simultaneous(ylabel= "MultiComparison", xlabel= "Score")
        plt.show()
    print(post_hoc_res.summary())
    print("----------------------------------------------------------------------------------------")
    #print("---------------------------------PAIRWISE COMPARISONS-----------------------------------")
    print("-----------------------------------------H4---------------------------------------------")
    interaction_groups = "Hyper_" + dataf.h1.astype(str)
    comp = mc.MultiComparison(dataf["score"], interaction_groups)
    post_hoc_res = comp.tukeyhsd()
    if argv.visu==0:
        post_hoc_res.plot_simultaneous(ylabel= "MultiComparison", xlabel= "Score")
        plt.show()
    print(post_hoc_res.summary())

def melted_pairwise(argv,dataf):
    print("----------------------------------------------------------------------------------------")
    print("---------------------------------PAIRWISE COMPARISONS-----------------------------------")
    print("----------------------------------------------------------------------------------------")
    interaction_groups = "Hyper_" + dataf.hyper.astype(str) + " & " + "Prob_" + dataf.prob.astype(str)
    comp = mc.MultiComparison(dataf["score"], interaction_groups)
    post_hoc_res = comp.tukeyhsd()
    if argv.visu==0:
        post_hoc_res.plot_simultaneous(ylabel= "MultiComparison", xlabel= "Score")
        plt.show()
    print(post_hoc_res.summary())

def main(argv):
    if argv.melted == 0:
        dataf,formula = two_factor_version(argv)
    elif argv.melted == 1:
        dataf,formula = four_factor_version()
    else:
        packagetest()
        exit()
    res = stat()
    print("-----------------------------------------------------------------------------------------")
    print("--------------------------------------POISSON MODEL--------------------------------------")
    print("-----------------------------------------------------------------------------------------")
    model = poisson(formula, data=dataf).fit()
    print(model.summary())
    print("-----------------------------------------------------------------------------------------")
    print("----------------------------------------OLS MODEL----------------------------------------")
    print("-----------------------------------------------------------------------------------------")
    model = ols(formula, data=dataf).fit()
    print(model.summary())
    print("-----------------------------------------------------------------------------------------")
    print("---------------------------------------ANOVA TABLE---------------------------------------")
    print("-----------------------------------------------------------------------------------------")
    anova_table = anova_lm(model, typ=3)
    print(anova_table)
    print("-----------------------------------------------------------------------------------------")
    print("--------------------------------------ANOVA SUMMARY--------------------------------------")
    print("-----------------------------------------------------------------------------------------")
    res.anova_stat(df=dataf, res_var='score', anova_model=formula)
    print(res.anova_summary)
    if argv.melted == 0:
        melted_pairwise(argv,dataf)
    elif argv.melted == 1:
        pairwise2(argv,dataf)
        #pairwise(res,dataf,formula)
    if argv.visu==0:
        if argv.melted == 0:
            interaction_plot(x=dataf['prob'], trace=dataf['hyper'], response=dataf['score'])
            plt.title("interaction plot")
            plt.show()
        visu(res)
    if argv.reg == 0:
        w, pvalue = stats.shapiro(res.anova_model_out.resid)
        print(f"SHAPIRO: W:{w}, P-VALUE {pvalue}")
    if argv.melted == 0:
        print("BARTLETT")
        res.bartlett(df=dataf,res_var='score',xfac_var='hyper')
        print(res.bartlett_summary)
        print("LEVENE")
        res.levene(df=dataf, res_var='score', xfac_var='hyper')
        print(res.levene_summary)

def analysis():
    distr = np.random.dirichlet(np.ones(4),size=1)
    print(distr)

def calculate_rand(episodes=15):
    #df = pd.DataFrame()
    df = pd.DataFrame(columns=['score','h1','h2','h3','h4','const'])
    for i in range(episodes):
        val = float(random.randrange(1200,1600))
        hyp = np.random.dirichlet(np.ones(4),size=1)
        summing = sum(hyp[0])
        '''new_row = {'value': val,
                'h1': hyp[0][0],
                'h2':hyp[0][1],
                'h3':hyp[0][2],
                'h4':hyp[0][3]}'''
        row = [val,hyp[0][0],hyp[0][1],hyp[0][2],hyp[0][3],summing]
        print("ROW", i+1)
        row = np.array(row).astype(float)
        df.loc[len(df)] = row
    return df

def calculate():
    dir = "/Users/antoniomone/Desktop/Uni/LEIDEN/THESIS/WANN/brain-tokyo-workshop/WANNRelease/prettyNeatWann/p/otherAtariSizes"
    entries = os.scandir(dir)
    for i in entries:
        print(i)
        
def calc_man():
    df = pd.DataFrame(columns=['index','score','h1','h2','h3','h4'])
    df.loc[0] = [len(df),1238.165,0.50,0.25,0.20,0.05]#def  1088.33+1390    #18 --- 
    df.loc[1] = [len(df),1455.835,0.25,0.45,0.25,0.05]#mine 1271.67+1516    #9  *-- 1459.17 #5    *-- 1577.5    #2 *-- 1595.0
    df.loc[2] = [len(df),1365.0,0.25,0.25,0.25,0.25]#equal 1400+1330        #15 ---
    df.loc[3] = [len(df),1366.5,0.20,0.40,0.20,0.20]#rand1 1415+1318        #13 ---
    df.loc[4] = [len(df),1245.0,0.40,0.20,0.20,0.20]#rand2 1280+1210        #17 ---
    df.loc[5] = [len(df),1542.7075,0.20,0.20,0.40,0.20]#rand3 1590+1425        #5  *-- 1673.33  #1   *-- 1482.5    #4  ---
    df.loc[6] = [len(df),1482,0.20,0.20,0.20,0.40]#ran4 1561.67+1228    #8  *-- 1490.0   #2   *-- 1648.33  #1  *-- 1611.67
    df.loc[7] = [len(df),1330.0,0.30,0.30,0.30,0.10]#rand5 1440+1220        #16 ---
    
    df.loc[8] = [len(df),1520.835,0.30,0.30,0.10,0.30]#rand6 1466.67+1575   #3  *-- 1360.0   #9   ---
    df.loc[9] = [len(df),1443.335,0.30,0.10,0.30,0.30]#rand7 1211.67+1675   #7  *-- 1187.50  #10  ---
    df.loc[10] = [len(df),1470.667,0.10,0.30,0.30,0.30]#rand8 1560+1356       #6  *-- 1424.17  #6   *-- 1542.5    #3  ---
    df.loc[11] = [len(df),1381.665,0.40,0.25,0.25,0.10]#ran9 1228.33+1535   #11 ---
    df.loc[12] = [len(df),1835.335,0.10,0.40,0.25,0.25]#ran10 1925+1781.67  #1  *-- 1487.50  #3   *-- 1267.5    #6  ---
    df.loc[13] = [len(df),1232.5,0.25,0.10,0.40,0.25]#ran11 1250+1215       #19 ---
    df.loc[14] = [len(df),1530.835,0.25,0.25,0.10,0.40]#ra12 1206.67+1855   #2  *-- 1398.33  #7   ---
    
    df.loc[15] = [len(df),1377.5,0.55,0.15,0.15,0.15]#ran13 1335+1420       #12 ---
    df.loc[16] = [len(df),1365.835,0.15,0.55,0.15,0.15]#ran14 1415+1316.67  #14 ---
    df.loc[17] = [len(df),1519.165,0.15,0.15,0.55,0.15]#ran15 1530+1508.33  #4  *-- 1370.83  #8   ---
    df.loc[18] = [len(df),1409.375,0.15,0.15,0.15,0.55]#ran16 1480+1290       #10 *-- 1475.0   #4   *-- 1392.5    #5  ---
    #print(df.head())
    #df.to_csv('/Users/antoniomone/Desktop/Uni/LEIDEN/THESIS/RLtests/hypcsv.csv')
    #onlyscore = df['score']
    #onlyscore.to_csv('/Users/antoniomone/Desktop/Uni/LEIDEN/THESIS/RLtests/score_hyp.csv')
    #hyps = df[['h1','h2','h3','h4']]
    #hyps.to_csv('/Users/antoniomone/Desktop/Uni/LEIDEN/THESIS/RLtests/hyps_hyp.csv')
    #print("csv written")
    return df

def swap_rows(df, row1, row2):
    df.iloc[row1], df.iloc[row2] =  df.iloc[row2].copy(), df.iloc[row1].copy()
    return df