import numpy as np, pandas as pd
def crm_df(seed=4, n=400):
    rng=np.random.default_rng(seed)
    stages=["Prospecting","Qualification","Proposal","Negotiation","Closed Won","Closed Lost"]
    reps={"Asha":300000,"Ben":250000,"Chen":250000,"Dana":200000,"Eli":200000}
    rows=[]
    asof=pd.Timestamp("2025-09-30")
    for i in range(n):
        rep=rng.choice(list(reps)); st=rng.choice(stages,p=[.15,.15,.15,.1,.25,.2])
        created=asof-pd.Timedelta(days=int(rng.integers(5,240)))
        close=created+pd.Timedelta(days=int(rng.integers(20,150)))
        amt=float(rng.lognormal(10,0.8)); prob={"Prospecting":10,"Qualification":20,"Proposal":50,"Negotiation":75,"Closed Won":100,"Closed Lost":0}[st]
        last=min(asof, created+pd.Timedelta(days=int(rng.integers(0,200))))
        rows.append({"Opportunity Name":f"Deal {i}","Account":f"Acme {i%120}","Owner":rep,"Stage":st,"Amount":round(amt,2),
                     "Probability (%)":prob,"Created Date":created,"Close Date":close,"Last Activity Date":last,
                     "Next Step":rng.choice(["Send proposal","Call CFO","", None],p=[.4,.3,.15,.15]),"Quota":reps[rep],
                     "Forecast Category":{"Negotiation":"Commit","Proposal":"Best Case"}.get(st,"Pipeline" if "Closed" not in st else "Closed")})
    return pd.DataFrame(rows)
