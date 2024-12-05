'''
# APP para la visualización de detalles del álbum de un artista
# Diseñada utilizando dashtools
# Buscar un artista, seleccionar uno de sus álbumes y esperar
# Fin: facilitar al interesado información sobre un proyecto
# La app acude a dos APIs diferentes, puede haber problemas de asociación
'''

import time
import dash
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, dcc, html, State, ALL, ctx
import matplotlib
import requests
from lyricsgenius import Genius
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import re
import base64
import networkx as nx

# Configuración de Genius API
GENIUS_API_TOKEN = "Q_qYMmOc6wMyzbHgrDZWm0vJAseysPCy9xD4QtYy1CuaN4AIHIByutJIt3Z7H4gD"
genius = Genius(GENIUS_API_TOKEN, remove_section_headers=True)

# Configuración de Spotify API
CLIENT_ID = 'a5dc6539854b4e2393869570c4b4f56a'
CLIENT_SECRET = '2609093d425249a0aa9445ac10f81511'

AUTH_URL = 'https://accounts.spotify.com/api/token'
BASE_URL = 'https://api.spotify.com/v1/'

# POST
auth_response = requests.post(AUTH_URL, {
    'grant_type': 'client_credentials',
    'client_id': CLIENT_ID,
    'client_secret': CLIENT_SECRET,
})

# Token
auth_response_data = auth_response.json()
spotify_token = auth_response_data['access_token']

# Analizador de sentimiento
sentiment_analyzer = SentimentIntensityAnalyzer()

# Configuración necesaria
matplotlib.use('Agg')

# App Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layouts
app.layout = dbc.Container([
    html.Div(style={'backgroundColor': '#DFFFD6', 'minHeight': '100vh', 'padding': '20px'}, children=[
        dbc.Row([
            dbc.Col([
                html.H1("Análisis de la obra de un artista", className="text-center mb-4", style={'color': '#2E8B57'}),
            ])
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Input(
                    id='artist-name-input',
                    type='text',
                    placeholder='Introduce el nombre del artista (preferiblemente angloparlante)',
                    className="form-control mb-3",
                    style={'borderRadius': '10px', 'border': '2px solid #2E8B57'}
                ),
                dbc.Button(
                    "Buscar artista",
                    id='search-artist-btn',
                    color="success",
                    className="mb-3",
                    style={'width': '100%'}
                ),
                html.Div(
                    id='artist-id-output',
                    className="text-muted mb-3 text-center",
                    style={
                        'minHeight': '30px',
                        'marginBottom': '10px'
                    }
                )
            ], width=6)
        ], justify='center'),
        dbc.Row([
            dbc.Col([
                html.Div(
                    id='album-list-output',
                    style={
                        'display': 'grid',
                        'gridTemplateColumns': 'repeat(8, 1fr)',
                        'gap': '10px',
                        'justifyItems': 'center',
                        'minHeight': '150px',
                        'marginBottom': '20px'
                    }
                )
            ], width=8),
            dbc.Col([
                html.Div([
                    html.H5(
                        id='selected-album-name',
                        className="text-center mb-2",
                        style={
                            'color': '#2E8B57',
                            'minHeight': '30px',
                            'marginBottom': '10px'
                        }
                    ),
                    html.Div(
                        id='album-image-url-output',
                        style={
                            'minHeight': '200px',
                            'display': 'flex',
                            'alignItems': 'center',
                            'justifyContent': 'center',
                            'border': '2px solid #2E8B57',
                            'borderRadius': '10px',
                            'padding': '8px',
                            'width': '60%',
                            'aspectRatio': '1 / 1',
                            'margin': '0 auto'
                        }
                    )
                ])
            ], width=4)
        ]),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H3(
                        "Análisis de sentimiento",
                        className="text-center mt-4 mb-4",
                        style={'color': '#2E8B57'}
                    ),
                    dcc.Graph(
                        id='sentiment-graph',
                        style={'margin': '0 auto', 'width': '80%', 'marginBottom': '20px'}
                    )
                ])
            ])
        ]),
        dbc.Row([
            dbc.Col([
                html.Div([
                    dcc.Graph(
                        id='sentiment-graph-normalized',
                        style={'margin': '0 auto', 'width': '80%', 'marginBottom': '20px'}
                    )
                ])
            ])
        ]),
        dbc.Row([
            dbc.Col([
                html.Div([
                    dcc.Graph(
                        id='sentiment-graph-bigotes',
                        style={'margin': '0 auto', 'width': '80%', 'marginBottom': '20px'}
                    )
                ])
            ])
        ]),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H3(
                        "Mapa de palabras del álbum",
                        className="text-center mt-4 mb-4",
                        style={'color': '#2E8B57'}
                    ),
                    html.Div(
                    id='wordcloud-output',
                    style={'textAlign': 'center', 'marginBottom': '20px'}
                    )
                ])
            ])
        ]),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H3(
                        "Popularidad de las canciones",
                        className="text-center mt-4 mb-4",
                        style={'color': '#2E8B57'}
                    ),
                    dcc.Graph(
                        id='popularity-graph',
                        style={'margin': '0 auto', 'width': '80%', 'marginBottom': '20px'}
                    )
                ])
            ])
        ]),
        # Almacenamiento y gráfico de red
        dbc.Row([
            # Almacenamiento invisible para map_features
            dcc.Store(id='map-features-storage', data={}),
            
            # Gráfico de red interactivo
            dbc.Col([
                html.Div([
                    html.H3(
                        "Mapa de red de artistas",
                        className="text-center mt-4 mb-4",
                        style={'color': '#2E8B57'}
                    ),
                    dcc.Graph(
                        id='network-graph',
                        style={'height': '600px', 'margin': '0 auto', 'width': '80%'}
                    )
                ])
            ])
        ])
    ])
], fluid=True)

# Callback para buscar álbumes
@app.callback(
    [Output('artist-id-output', 'children'),
    Output('album-list-output', 'children')],
    Input('search-artist-btn', 'n_clicks'),
    State('artist-name-input', 'value')
)
def get_artist_albums(n_clicks, artist_name):
    if not n_clicks or not artist_name:
        return '', ''

    search_url = f"https://api.genius.com/search?q={artist_name}" # Lo busca en la API de Genius
    headers = {"Authorization": f"Bearer {GENIUS_API_TOKEN}"}
    response = requests.get(search_url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        artist_id = next((hit['result']['primary_artist']['id']
                          for hit in data['response']['hits']
                          if hit['result']['primary_artist']['name'].lower() == artist_name.lower()), None)
        if not artist_id:
            return "No se encontró el artista.", ''

        albums = genius.artist_albums(artist_id, per_page=50)['albums'] # Buscamos sus álbumes y creamos los botones
        album_buttons = [
            html.Div([
                html.Img(
                    src=album.get('cover_art_url', "https://via.placeholder.com/150"),
                    style={
                        'width': '100px',
                        'height': '100px',
                        'borderRadius': '10px',
                        'border': f"2px solid #2E8B57",
                        'cursor': 'pointer',
                        'objectFit': 'cover'
                    },
                    id={'type': 'album-cover', 'index': album['id']}
                )
            ])
            for album in albums
        ]
        return f"El ID del artista '{artist_name}' es: {artist_id}. Tras seleccionar un álbum, espera 1-2min.", album_buttons # ID de Genius (trazabilidad)
    else:
        return '', "Error al consultar la API de Genius."

# Callback para selección, análisis y generación de varias gráficas
@app.callback(
    [Output('album-image-url-output', 'children'),
    Output('selected-album-name', 'children'),
    Output('sentiment-graph', 'figure'),
    Output('sentiment-graph-normalized', 'figure'),
    Output('sentiment-graph-bigotes', 'figure'),
    Output('wordcloud-output', 'children'),
    Output({'type': 'album-cover', 'index': ALL}, 'style'),
    Output('map-features-storage', 'data')],
    [Input({'type': 'album-cover', 'index': ALL}, 'n_clicks')],
    [State({'type': 'album-cover', 'index': ALL}, 'id')]
)
def analyze_album(n_clicks, album_ids):
    # Estilos de los álbumes
    default_style = {
        'width': '100px',
        'height': '100px',
        'borderRadius': '10px',
        'border': '2px solid #2E8B57',
        'cursor': 'pointer',
        'objectFit': 'cover'
    }
    selected_style = default_style.copy()
    selected_style['border'] = '4px solid #FFD700'  # Seleccionado en amarillo

    # Mientras no se selecciona ninguno
    if not n_clicks or all(click is None for click in n_clicks):
        num_albums = len(album_ids)
        return (
            html.Div("Selecciona un álbum"),
            "",
            go.Figure(),
            go.Figure(),
            go.Figure(),
            html.Div("No hay mapa de palabras disponible."),
            [default_style] * num_albums,
            []
        )

    # Detectar el álbum seleccionado
    triggered_id = ctx.triggered_id
    if not triggered_id:
        num_albums = len(album_ids)
        return (
            html.Div("Selecciona un álbum"),
            "",
            go.Figure(),
            go.Figure(),
            go.Figure(),
            html.Div("No hay mapa de palabras disponible."),
            [default_style] * num_albums,
            []
        )

    # Extraer su ID
    album_id = triggered_id['index']

    try:
        # Info del álbum
        album = genius.album(album_id)['album']
        cover_art = album.get('cover_art_url', "https://via.placeholder.com/300")
        album_name = album.get('name', "Álbum desconocido")

        # Tracks del álbum
        tracks = genius.album_tracks(album_id)['tracks']
        if not tracks:
            raise ValueError("No se encontraron canciones en este álbum.")

        # Análisis de sentimiento
        song_titles = []
        positive_scores, neutral_scores, negative_scores = [], [], []

        all_lyrics = ""

        map_features = []

        for track in tracks:
            song = track.get('song', {})
            song_title = song.get('title', "Título no disponible")
            song_id = song.get('id')

            main_artists = [song['primary_artist']['name']]  # Artista principal
            feat_artists = [artist['name'] for artist in song.get('featured_artists', [])]

            # Agregar al mapa de features (aprovechamos este proceso)
            map_features.append({
                'song_title': song_title,
                'main_artists': main_artists,
                'feat_artists': feat_artists
            })

            if not song_id:
                continue

            # Extraer letras
            lyrics = genius.lyrics(song_id=song_id, remove_section_headers=True)
            if not lyrics:
                continue

            # Limpieza de las letras
            lyrics = clean_song_lyrics(lyrics)
            all_lyrics += " " + lyrics

            # Análisis de sentimiento
            scores = sentiment_analyzer.polarity_scores(lyrics)
            positive_scores.append(scores['pos'])
            neutral_scores.append(scores['neu'])
            negative_scores.append(scores['neg'])
            song_titles.append(song_title)

        # Normalización para segunda visualización (para comparación interna del álbum)
        def normalize(scores):
            max_val, min_val = max(scores, default=0), min(scores, default=0)
            return [(s - min_val) / (max_val - min_val) if max_val > min_val else 0 for s in scores]

        n_positive_scores = normalize(positive_scores)
        n_neutral_scores = normalize(neutral_scores)
        n_negative_scores = normalize(negative_scores)

        # Creación del gráfico de sentimiento
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=song_titles, y=positive_scores, mode='lines+markers', name='Positivo', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=song_titles, y=neutral_scores, mode='lines+markers', name='Neutral', line=dict(color='gray')))
        fig.add_trace(go.Scatter(x=song_titles, y=negative_scores, mode='lines+markers', name='Negativo', line=dict(color='red')))
        fig.update_layout(
            title="Análisis de sentimiento por canción",
            xaxis_title="Canciones",
            yaxis_title="Puntuación de sentimiento",
            template="plotly_white"
        )

        # Creación del gráfico de sentimiento normalizado
        fig_n = go.Figure()
        fig_n.add_trace(go.Scatter(x=song_titles, y=n_positive_scores, mode='lines+markers', name='Positivo', line=dict(color='green')))
        fig_n.add_trace(go.Scatter(x=song_titles, y=n_neutral_scores, mode='lines+markers', name='Neutral', line=dict(color='gray')))
        fig_n.add_trace(go.Scatter(x=song_titles, y=n_negative_scores, mode='lines+markers', name='Negativo', line=dict(color='red')))
        fig_n.update_layout(
            title="Análisis de sentimiento por canción (normalizado)",
            xaxis_title="Canciones",
            yaxis_title="Puntuación de sentimiento",
            template="plotly_white"
        )

        # Creación del gráfico de bigotes
        box_fig = go.Figure()

        # Añadir los datos para cada emoción
        box_fig.add_trace(go.Box(x=positive_scores, name="Positivo", marker=dict(color='green'), orientation='h'))
        box_fig.add_trace(go.Box(x=neutral_scores, name="Neutral", marker=dict(color='gray'), orientation='h'))
        box_fig.add_trace(go.Box(x=negative_scores, name="Negativo", marker=dict(color='red'), orientation='h'))

        # Configuración del layout
        box_fig.update_layout(
            title="Distribución de sentimientos del álbum",
            xaxis_title="Puntuación de sentimiento",
            yaxis_title="Emoción",
            template="plotly_white",
            showlegend=False
        )

        # Generación del wordcloud
        wordcloud_image = generate_wordcloud_image(all_lyrics)

        # Actualizamos los estilos dinámicamente
        styles = [
            selected_style if album_ids[i]['index'] == album_id else default_style
            for i in range(len(album_ids))
        ]

        return (
            html.Img(src=cover_art, style={'width': '100%', 'height': 'auto', 'borderRadius': '10px'}),
            album_name,
            fig,
            fig_n,
            box_fig,
            html.Img(src=wordcloud_image, style={'width': '80%', 'height': 'auto'}),
            styles,
            map_features
        )

    except Exception as e:
        print(f"Error fetching album: {e}")
        num_albums = len(album_ids)
        return (
            html.Div("Error al obtener el álbum"),
            "Error",
            go.Figure(),
            go.Figure(),
            go.Figure(),
            html.Div("No se pudo generar el mapa de palabras."),
            [default_style] * num_albums,
            []
        )

def clean_song_lyrics(raw_lyrics):
    """
    Limpia las letras eliminando los encabezados iniciales hasta después de 'Lyrics',
    eliminando las etiquetas finales como '23Embed', '4Embed', etc., y convirtiendo todo a minúsculas.
    """
    # Identificar y eliminar todo hasta después de "Lyrics"
    if "Lyrics" in raw_lyrics:
        start_index = raw_lyrics.find("Lyrics") + len("Lyrics")
        raw_lyrics = raw_lyrics[start_index:]

    # Identificar y eliminar todo lo que corresponde a "XXEmbed" al final
    clean_lyrics = re.sub(r"\d+Embed$", "", raw_lyrics).strip()

    # Minúsculas (para wordmap)
    clean_lyrics = clean_lyrics.lower()

    return clean_lyrics

def generate_wordcloud_image(lyrics):

    # Las stopwords automáticas SON EN INGLÉS
    wordcloud = WordCloud(
        width=2000,
        height=1000,
        background_color='white',
        stopwords=None,
        min_word_length=4
    ).generate(lyrics)

    # Hay que exportarlo como un BytesIO
    buffer = io.BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=2)
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)

    # Convertimos a cadena base64 (única forma que he encontrado)
    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{base64_image}"

# Función para buscar álbum en la API DE SPOTIFY
def search_album(artist_name, album_name, token):
    headers = {
        'Authorization': f'Bearer {token}'
    }
    search_url = f'{BASE_URL}search'
    params = {
        'q': f'album:{album_name} artist:{artist_name}',
        'type': 'album',
        'limit': 1  # Solo el más relevante
    }
    response = requests.get(search_url, headers=headers, params=params)
    data = response.json()
    albums = data.get('albums', {}).get('items', [])
    if albums:
        return albums[0]['id'], albums[0]['name']
    else:
        return None, None

# Obtener las tracks en la API DE SPOTIFY
def get_album_tracks(album_id, token):
    headers = {
        'Authorization': f'Bearer {token}'
    }
    album_tracks_url = f'{BASE_URL}albums/{album_id}/tracks'
    response = requests.get(album_tracks_url, headers=headers)
    data = response.json()
    tracks = data.get('items', [])
    return [(track['id'], track['name']) for track in tracks]

# Obtener la popularidad en la API DE SPOTIFY
def get_track_popularity(track_ids, token):
    headers = {
        'Authorization': f'Bearer {token}'
    }
    track_url = f"{BASE_URL}tracks"
    popularity_data = []
    for track_id in track_ids:
        response = requests.get(f"{track_url}/{track_id}", headers=headers)
        track_info = response.json()
        popularity_data.append((track_info['name'], track_info['popularity']))
    return popularity_data

# Representamos la popularidad en un gráfico
def plot_popularity(popularity_data):
    track_names, popularity_scores = zip(*popularity_data)

    fig = go.Figure(
        [go.Bar(x=track_names,
                y=popularity_scores,
                text=popularity_scores,
                textposition='auto',
                marker=dict(color='#2E8B57'))]
    )
    fig.update_layout(
        title="Popularidad de las canciones",
        xaxis_title="Canciones",
        yaxis_title="Puntuación",
        template="plotly_white"
    )
    return fig

# Callback para la actualización de la popularidad con el gráfico
@app.callback(
    Output('popularity-graph', 'figure'),
    State('artist-name-input', 'value'),
    Input('selected-album-name', 'children')
)
def update_popularity_chart(artist_name, album_name):
    if not album_name or not artist_name:
        return go.Figure()

    album_id, _ = search_album(artist_name, album_name, spotify_token)
    if not album_id:
        return go.Figure()

    track_data = get_album_tracks(album_id, spotify_token)
    track_ids, track_names = zip(*track_data)
    popularity_data = get_track_popularity(track_ids, spotify_token)

    return plot_popularity(popularity_data)

# Callback para la actualización de la red de artistas
@app.callback(
    Output('network-graph', 'figure'),
    Input('map-features-storage', 'data') # Entrada desde el almacenamiento
)
def update_network_graph(map_features):
    if not map_features:
        # Devuelve un gráfico vacío si no hay datos
        return go.Figure()

    # Procesar los datos para nodos y enlaces
    nodes, edges = process_song_artist_data(map_features)

    # Crear la visualización interactiva
    return visualize_network(nodes, edges, map_features)

def process_song_artist_data(map_features):
    """
    Procesa los datos de canciones y artistas para construir una red.
    """
    nodes = set()
    edges = []

    for song in map_features:
        song_title = song['song_title']
        main_artists = song['main_artists']
        feat_artists = song['feat_artists']

        # Agregar la canción como nodo
        nodes.add(song_title)

        # Conectar la canción con sus artistas principales y destacados
        for artist in main_artists + feat_artists:
            nodes.add(artist)  # Agregar artistas como nodos
            edges.append((song_title, artist))  # Conectar canción y artista

    return nodes, edges

def visualize_network(nodes, edges, map_features):

    # Crear el gráfico
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Posiciones de nodos
    pos = nx.spring_layout(G)  # Algoritmo de distribución de nodos

    # Crear trazados de nodos y enlaces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x = []
    node_y = []
    node_color = [] # Colores de nodos
    node_text = []

    # Clasificar nodos según su tipo usando map_features
    song_titles = {feature['song_title'] for feature in map_features}
    main_artists = {artist for feature in map_features for artist in feature['main_artists']}
    feat_artists = {artist for feature in map_features for artist in feature['feat_artists']}

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        # Determinar el color del nodo
        if node in song_titles:
            node_color.append('#2E8B57') # Verde para canciones
        elif node in main_artists:
            node_color.append('blue') # Azul para artistas principales
        elif node in feat_artists:
            node_color.append('yellow') # Amarillo para artistas feat
        else:
            node_color.append('gray') # Por si hay nodos desconocidos

        node_text.append(node) # Añade el texto del nodo

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            size=20,
            color=node_color, # Usa los colores personalizados
            line_width=4
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title="Red de canciones y artistas",
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=40, l=40, r=40, t=100),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)