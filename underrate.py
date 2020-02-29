"""Functions for loading/processing rating data"""

import pandas as pd
import scipy.sparse

def loadRatings(fname, dim=None):
    raw = pd.read_csv(fname, sep="\t", header=None,
                   names=["user", "movie", "rating"])
    dim = (raw["user"].max(), raw["movie"].max()) if (dim is None) else dim
    sparse = scipy.sparse.csr_matrix(
        (raw["rating"], (raw["user"]-1, raw["movie"]-1)),
        shape=dim
    )
    return pd.DataFrame.sparse.from_spmatrix(
        sparse,
        index=pd.Index(list(range(1,dim[0]+1)), name="user"),
        columns=pd.Index(list(range(1, dim[1]+1)), name="movie"),
    ).astype(pd.SparseDtype("float", scipy.nan))

def loadMovies(fname):
    raw = pd.read_csv("data/movies.txt", sep="\t", header=None, index_col=0,
                     names=["movie","title","unknown","action","adventure","animation",
                            "childrens", "comedy","crime","documentary","drama",
                            "fantasy", "filmnoir", "horror","musical","mystery",
                            "romance","scifi","thriller","war","western"]
                    ).sort_index()
    return raw.iloc[:,[0]], raw.iloc[:,1:]

def ratingStats(ratings):
    userStats = pd.DataFrame({
        "nrated": ratings.count(axis=1),
        "useravg": ratings.mean(axis=1)
    })
    movieStats = pd.DataFrame({
        "nratings": ratings.count(axis=0),
        "movieavg": ratings.mean(axis=0)
    })
    return userStats, movieStats