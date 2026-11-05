import numpy as np, pandas as pd
def marketing_df(seed=3):
    rng=np.random.default_rng(seed); rows=[]
    ch={"Google Ads":(0.035,0.06,1.2,90),"Meta":(0.012,0.025,0.6,70),"LinkedIn":(0.006,0.03,4.5,400),"Email":(0.08,0.05,0.05,60),"TikTok":(0.01,0.004,0.4,50)}
    for d in pd.date_range("2025-01-01","2025-06-30"):
        for c,(ctr,cvr,cpc,aov) in ch.items():
            for camp in range(3):
                imp=int(rng.integers(2000,20000)); clk=rng.binomial(imp,ctr); conv=rng.binomial(clk,cvr)
                rows.append({"Date":d,"Channel":c,"Campaign":f"{c[:3]}-{camp}","Impressions":imp,"Clicks":clk,
                             "Spend":round(clk*cpc*rng.uniform(.8,1.2),2),"Conversions":conv,"Revenue":round(conv*aov*rng.uniform(.7,1.3),2),
                             "CTR":round(clk/imp,4) if imp else 0})
    return pd.DataFrame(rows)
