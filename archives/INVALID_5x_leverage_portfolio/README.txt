INVALID checkpoints — entrainees sur portefeuille casse (levier fantome 5x).
Le run DiagGaussian (20260625_115926) etait stable mais le portfolio_manager
autorisait des positions a 5x le cash (ligne 743 "* 5.0"), produisant une
equity physiquement impossible (20$ -> 3792$). Les rewards PnL etaient donc
fausses. Corrige par commit c216b6d. NE PAS reprendre depuis ces checkpoints.
