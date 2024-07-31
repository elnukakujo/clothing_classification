import plotly.express as px

def display_image(index:0,x,y):
    fig = px.imshow(x[index], color_continuous_scale='gray')
    fig.update_layout(
        title='This is a {}'.format('T-shirt' if y[index] == 0 else 'Trouser'),
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
    )
    fig.show()