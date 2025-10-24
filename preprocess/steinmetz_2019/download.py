# Needs: pip install git+https://github.com/cognoma/figshare
from figshare.figshare import Figshare

fs = Figshare()
article_id = 9598406
fs.get_article_details(article_id)
fs.retrieve_files_from_article(article_id, "steinmetz2019")
