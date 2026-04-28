from scipy.stats import ttest_ind

def run_ttest(df):
    
    a = df[df['supplier'] == 'A']['demand']
    b = df[df['supplier'] == 'B']['demand']
    
    stat, p = ttest_ind(a, b)
    
    print("T-test p-value:", p)