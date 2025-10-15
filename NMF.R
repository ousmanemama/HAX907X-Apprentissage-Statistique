install.packages('NMF')
BiocManager::install("Biobase")
library(Biobase)

jouet=read.table("recom-jouet.txt")
jouet
boxplot(jouet)

library(NMF)
nmfAlgorithm()
nmfAlgorithm("brunet")
nmfAlgorithm("lee")
nmfAlgorithm("pe-nmf")
nmfAlgorithm("snmf/l")
nmfAlgorithm("snmf/r")
res.multi.method=nmf(jouet, 5,nrun=10,
                     list("brunet","lee","snmf/l","snmf/r"),
                     seed = 111, .options ="t")
compare(res.multi.method)
consensusmap(res.multi.method,hclustfun="ward")

## snmf/l semble le plus performant :
## blocs rouges bien définis,
## silhouette élevée (0.89).

estim.r=nmf(jouet,2:6,method="snmf/l",
            nrun=10,seed=111)
plot(estim.r)
consensusmap(estim.r)

##  méthode snmf/l   ##

nmf.jouet=nmf(jouet,4,method="snmf/l",
              nrun=30,seed=111)

summary(nmf.jouet)
# les matrices de facteurs
w=basis(nmf.jouet) #représentation d’un individu/ligne dans l’espace latent
h=coef(nmf.jouet) #contribution des facteurs aux colonnes originales
dim(w)
dim(h)

# Matrice reconstruite
xchap=w%*%h
xchap
# Comparer avec les données initiales
# Identifier le plus fort score reconstruit
# par client
prod=apply(xchap-10*jouet,1,function(x)
  which.max(x))
# Identifier le produit correspondant
dN.h=dimnames(h)[[2]]
dN.v=dimnames(w)[[1]]

cbind(dN.v,dN.h[prod])


jouet_mat <- as.matrix(jouet)

#  Représentation des classifications
basismap(nmf.jouet,hclustfun="ward")
coefmap(nmf.jouet,hclustfun="ward")







#######################SVD#####################################################
##############################################################################
library(FactoMineR)
PCA(jouet)
PCA(t(jouet))

# approximation de rang 2 par SVD
res=svd(jouet)
res 
# Matrice reconstruite
xchap_svd=res$u[,1:2]%*%diag(res$d[1:2])%*%t(res$v[,1:2])
xchap_svd
# Comparer avec les données initiales
# Identifier le plus fort score reconstruit
# par client
prod_svd=apply(xchap_svd-10*jouet,1,function(x)
  which.max(x))
# Identifier le produit correspondant
cbind(dN.v,dN.h[prod_svd])











