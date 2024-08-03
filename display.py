import plotly.express as px
import plotly.graph_objects as go

def display_image(x,y):
    fig = px.imshow(x, color_continuous_scale='gray')
    fig.update_layout(
        title='This is a {}'.format('T-shirt' if y == 0 else 'Trouser'),
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
    )
    fig.show()
    
def display_metrics(costs= False, train_accs= False, test_accs= False):
    fig=go.Figure()
    if costs: 
        fig.add_trace(go.Scatter(
            x=list(range(len(costs))),
            y=costs,
            mode='lines',
            name='Cost'
        ))

    if train_accs:
        fig.add_trace(go.Scatter(
            x=list(range(len(train_accs))),
            y=train_accs,
            mode='lines',
            name='Train Accuracy'
        ))
    
    if test_accs:
        fig.add_trace(go.Scatter(
            x=list(range(len(test_accs))),
            y=test_accs,
            mode='lines',
            name='Test Accuracy'
        ))

    fig.update_layout(
        title="Evolution of Cost, and Training/Test Accuracy over the Training Steps",
        xaxis_title="Training Step",
        yaxis_title="Value",
        legend_title="Metrics"
    )
    fig.show()