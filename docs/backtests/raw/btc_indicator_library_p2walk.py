import json, sys
import numpy as np
import btc_indicator_library as L
d=sys.argv[1]
F=np.load(d+"/feat.npz",allow_pickle=True); X,t15=F["X"],F["t"]; names=[str(x) for x in F["names"]]
O=np.load(d+"/outcomes.npz"); ln,sn,lx,sx,t5=O["ln"],O["sn"],O["lx"],O["sx"],O["t5"]
i20=int(np.searchsorted(t15,L.ts(2020)+60*L.DAY)); i23=int(np.searchsorted(t15,L.ts(2023))); i24=int(np.searchsorted(t15,L.ts(2024))); iend=len(t15)-2
Q=np.nanquantile(X[i20:i23],(0.1,0.25,0.75,0.9),axis=0)
bn=[n.split("_",1)[1] if n.split("_",1)[0] in ("15m","1h","4h","1d") else n for n in names]
R=json.load(open(d+"/p2_result.json"))
out=[]
for r in R["final3"]:
    ml=np.ones(len(t15),bool); ms=np.ones(len(t15),bool)
    for f,k in r["key"]:
        with np.errstate(invalid="ignore"):
            ml&= X[:,f]<=Q[k,f] if k<2 else X[:,f]>=Q[k,f]
        g,kk=f,k
        if bn[f] in L.DIRECTIONAL: kk=3-k
        elif bn[f] in ("pdi","mdi"): g=names.index(names[f].replace("pdi","mdi") if bn[f]=="pdi" else names[f].replace("mdi","pdi"))
        with np.errstate(invalid="ignore"):
            ms&= X[:,g]<=Q[kk,g] if kk<2 else X[:,g]>=Q[kk,g]
    el,es=ml&~np.r_[False,ml[:-1]], ms&~np.r_[False,ms[:-1]]
    row={"rule":r["rule"]}
    for per,a,b in (("train",i20,i23),("2023",i23,i24),("2024~",i24,iend)):
        tr=L.walk(el,es,ln,sn,lx,sx,t5,t15,a,b); yrs=(t15[b-1]-t15[a])/(365.25*L.DAY)
        row[per]=L.stats(tr,t15,1.0,yrs); row[per+"_3x"]=L.stats(tr,t15,3.0,yrs)
        row[per+"_cost15"]=L.stats(L.walk(el,es,ln,sn,lx,sx,t5,t15,a,b,0.001),t15,1.0,yrs)
        if per=="2024~":
            row["per_year"]={}
            for Y in (2024,2025,2026):
                ya,yb=int(np.searchsorted(t15,L.ts(Y))),min(int(np.searchsorted(t15,L.ts(Y+1))),iend)
                row["per_year"][Y]=L.stats(L.walk(el,es,ln,sn,lx,sx,t5,t15,ya,yb),t15)
            null=L.circ_null(el,es,ln,sn,lx,sx,t5,t15,a,b,n=1000)
            row["walk_null_mean"]=round(float(null.mean()*100),3); act=row[per]["ev"]/100
            row["walk_null_p"]=round(float((np.sum(null>=act)+1)/1001),4)
    out.append(row)
    print(json.dumps(row,default=float,ensure_ascii=False)[:1500])
json.dump(out,open(d+"/p2_walk.json","w"),default=float,ensure_ascii=False,indent=1)
