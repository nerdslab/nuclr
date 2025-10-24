# Needs: pip install git+https://github.com/cognoma/figshare
from figshare.figshare import Figshare

fs = Figshare()
article_id = 19448531
fs.retrieve_files_from_article(article_id, "Bugeon et al 2022 v2.zip")
