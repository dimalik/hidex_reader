High-dimensional Space Python Wrapper
============

A python interface for interacting with semantic space models of meaning.
Offers out of the box support for HiDEx (Shaoul & Westbury 2006; 2010) and S-Space models (https://github.com/fozziethebeat/S-Space) and is easily extendable to any other |V| x |D| matrix.

Initialise a model with e.g.:
> model = HidexModel(PATH_TO_DICT, PATH_TO_GCM)

or

> model = SSpaceModel(PATH_TO_SSPACE)

PATH_TO_DICT should point to the .dict file output by HiDEx (e.g. combined.dict) and PATH_TO_GCM to the corresponding .gcm file.
In the case of S-Space, PATH_TO_SSPACE should point to the .sspace file which has to be saved using TEXT (i.e. not binary) format.

You can perform various word tasks with the model. Some of them are already built-in:

> model.get_similarity('dog', 'cat') <- cosine similarity

> model.get_neighbourhood('dog', topn=N) <- closest N neighbours using cosine
    similarity
    
> model.get_arc('dog') <- get neighbourhood size/density
