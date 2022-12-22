from pandas import DataFrame
from typing import List
from IPython.display import display_html

def mdisplay(dfs: List[DataFrame], names:List[str]=[]):
    """
    Displays several data frames side by side
    
    Adapded form
    https://stackoverflow.com/questions/38783027/jupyter-notebook-display-two-pandas-tables-side-by-side
    """

    html_str = ''
    if names:
        html_str += ('<tr>' +
                     ''.join(f'<td style="text-align:center">{name}</td>' for name in names) +
                     '</tr>')
    html_str += ('<tr>' +
                 ''.join(f'<td style="vertical-align:top"> {df.to_html(index=False)}</td>'
                         for df in dfs) +
                 '</tr>')
    html_str = f'<table>{html_str}</table>'
    html_str = html_str.replace('table','table style="display:inline"')
    display_html(html_str, raw=True)
