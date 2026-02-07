from typing import List, Optional
import plotly.graph_objects as go
from plotly.graph_objs import Figure

def plot_data(x: List[float],
              y: List[float],
              fig: Optional[Figure] = None,
              mode: str = 'lines+markers',
              name: str = 'Data',
              color: Optional[str] = None,
              size: Optional[int] = None,
              opacity: float = 0.8,
              show: bool = False,
              line_width: int = 3,  # исправлено на width
              line_dash: str = "solid",
              title: str = 'График данных',
              xlabel: str = 'X',
              ylabel: str = 'Y') -> Figure:
    """
    Функция для построения графика данных

    Parameters:
    -----------
    x : List[float]
        Список значений по оси X
    y : List[float]
        Список значений по оси Y
    fig : Optional[Figure], default=None
        Существующая фигура Plotly. Если None, создается новая
    mode : str, default='lines+markers'
        Режим отображения: 'markers', 'lines', 'lines+markers'
    name : str, default='Data'
        Имя данных для легенды
    color : Optional[str], default=None
        Цвет графика. Если None, используется цвет по умолчанию
    size : Optional[int], default=None
        Размер маркеров (только для режима с маркерами)
    opacity : float, default=0.8
        Прозрачность графика (0-1)
    show : bool, default=False
        Показать график сразу после создания
    line_width : int, default=3
        Толщина линии (если mode содержит 'lines')
    line_dash : str, default='solid'
        Стиль линии: 'solid', 'dash', 'dot', 'dashdot'
    title : str, default='График данных'
        Заголовок графика
    xlabel : str, default='X'
        Подпись оси X
    ylabel : str, default='Y'
        Подпись оси Y

    Returns:
    --------
    Figure
        Объект графика Plotly
    """
    if fig is None:
        fig = go.Figure()

    marker_config = {}
    if color:
        marker_config['color'] = color
    if size and ('markers' in mode):
        marker_config['size'] = size
    if opacity:
        marker_config['opacity'] = opacity

    line_config = None
    if 'lines' in mode:
        line_config = dict(width=line_width, dash=line_dash)
        if color:
            line_config['color'] = color

    # Добавляем график
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode=mode,
        name=name,
        marker=marker_config if marker_config else None,
        line=line_config,
        opacity=opacity
    ))

    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        showlegend=True,
        template='plotly_white'
    )

    if show:
        fig.show()

    return fig