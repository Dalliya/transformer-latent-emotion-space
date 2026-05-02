import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, BertForSequenceClassification
import umap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
import textwrap
import warnings
import html

# Suppress UMAP parallelization warnings for clean terminal execution
warnings.filterwarnings("ignore", category=UserWarning, module="umap")

# Dictionary mapping 28 classes to their actual emotion names
EMOTION_NAMES = {
    0: "admiration", 1: "amusement", 2: "anger", 3: "annoyance", 4: "approval",
    5: "caring", 6: "confusion", 7: "curiosity", 8: "desire", 9: "disappointment",
    10: "disapproval", 11: "disgust", 12: "embarrassment", 13: "excitement",
    14: "fear", 15: "gratitude", 16: "grief", 17: "joy", 18: "love", 19: "nervousness",
    20: "optimism", 21: "pride", 22: "realization", 23: "relief", 24: "remorse",
    25: "sadness", 26: "surprise", 27: "neutral"
}

# Custom Neon Palette for 28 Emotions
# Pleasant/Positive = Beautiful bright colors. Negative/Bad = Dirty, harsh, or dark colors.
CUSTOM_EMOTION_COLORS = {
    "Admiration": "#00FFFF",      # Bright Cyan
    "Amusement": "#FFD700",       # Gold
    "Anger": "#8B0000",           # Dark Red
    "Annoyance": "#B8860B",       # Dark Goldenrod (Dirty)
    "Approval": "#00FA9A",        # Medium Spring Green
    "Caring": "#FF69B4",          # Hot Pink
    "Confusion": "#808080",       # Gray
    "Curiosity": "#BA55D3",       # Medium Orchid
    "Desire": "#FF1493",          # Deep Pink
    "Disappointment": "#556B2F",  # Dark Olive Green (Dirty)
    "Disapproval": "#8B4513",     # Saddle Brown (Dirty)
    "Disgust": "#808000",         # Olive (Dirty)
    "Embarrassment": "#CD5C5C",   # Indian Red (Dull)
    "Excitement": "#00BFFF",      # Deep Sky Blue
    "Fear": "#4B0082",            # Indigo (Dark)
    "Gratitude": "#32CD32",       # Lime Green
    "Grief": "#2F4F4F",           # Dark Slate Gray (Gloomy)
    "Joy": "#FFFF00",             # Pure Yellow
    "Love": "#FFB6C1",            # Light Pink
    "Nervousness": "#D2691E",     # Chocolate/Orange-brown
    "Optimism": "#7FFFD4",        # Aquamarine
    "Pride": "#9370DB",           # Medium Purple
    "Realization": "#4682B4",     # Steel Blue
    "Relief": "#20B2AA",          # Light Sea Green
    "Remorse": "#A0522D",         # Sienna (Dirty brown)
    "Sadness": "#483D8B",         # Dark Slate Blue (Gloomy)
    "Surprise": "#FFA500",        # Orange
    "Neutral": "#A9A9A9"          # Dark Gray
}

def extract_embeddings(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """
    Extracts 768-dimensional embeddings from the BERT latent space.
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Extracting embeddings using device: {device}")

    model_name = "logasanjeev/bert-emotion-classifier"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, output_hidden_states=True).to(device)
    model.eval()

    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embeddings = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
            embeddings.append(cls_embeddings)

    return np.vstack(embeddings)

def clean_and_wrap_text(text: str) -> str:
    """Unescapes HTML entities (like &#x27;) and wraps text for tooltips."""
    clean_text = html.unescape(text)
    wrapped = '<br>'.join(textwrap.wrap(clean_text, width=50))
    return wrapped[:300] + ('...' if len(wrapped) > 300 else '')

def create_neon_comparative_umap(embeddings: np.ndarray, df: pd.DataFrame):
    """
    Generates a 2D side-by-side UMAP comparison dashboard with Matrix-style neon aesthetics.
    """
    print("Fitting 2D UMAP Manifold...")
    
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)

    df['UMAP_X'] = embeddings_2d[:, 0]
    df['UMAP_Y'] = embeddings_2d[:, 1]
    
    # Format labels
    df['Emotion_Class'] = df['predicted_28_classes'].map(EMOTION_NAMES).str.capitalize()
    df['Human_Polarity'] = df['label'].map({0: 'Negative', 1: 'Positive'})
    df['Machine_Polarity'] = df['predicted_label'].map({0: 'Negative', 1: 'Positive'})
    
    # Clean text to remove dirty encoding artifacts
    df['Hover_Text'] = df['text'].apply(clean_and_wrap_text)

    # Neon Color Palette: Matrix Green (Positive) and Cyber Red (Negative)
    neon_colors = {
        'Positive': '#39FF14', # Matrix Neon Green
        'Negative': '#FF003C'  # Cyber Neon Red
    }

    # Glowing Matrix Green Subplot Titles
    fig = make_subplots(
        rows=1, cols=2, 
        subplot_titles=(
            "<span style='color:#39FF14; font-size:15px; font-family: \"Courier New\", Courier, monospace; text-shadow: 0px 0px 8px #39FF14;'>Human Annotation (IMDB Ground Truth)</span>", 
            "<span style='color:#39FF14; font-size:15px; font-family: \"Courier New\", Courier, monospace; text-shadow: 0px 0px 8px #39FF14;'>BERT Classifier Mapping (Zero-Shot 28->2)</span>"
        ),
        horizontal_spacing=0.06
    )

    def build_hover_text(row, polarity_col_to_display):
        """Constructs the interactive tooltip with dynamic coloring."""
        em_color = CUSTOM_EMOTION_COLORS.get(row['Emotion_Class'], '#FFFFFF')
        mach_color = neon_colors[row['Machine_Polarity']]
        hum_color = neon_colors[row['Human_Polarity']]
        
        # Now the entire "Detected Emotion:" line matches the emotion color
        return (
            f"<span style='color:{em_color}'><b>Detected Emotion: {row['Emotion_Class']}</b></span><br><br>"
            f"<span style='color:{mach_color}'><b>Machine Interpreted As: {row['Machine_Polarity']}</b></span><br>"
            f"<span style='color:{hum_color}'><b>Human Annotated As: {row['Human_Polarity']}</b></span><br><br>"
            f"<span style='color:#FFFFFF'><i>Review Text:</i><br>{row['Hover_Text']}</span>"
        )

    # Plot traces grouped strictly by the 2 Polarity classes
    for polarity in ['Negative', 'Positive']:
        
        # --- LEFT PANEL: Colored by what the HUMAN annotated ---
        subset_gt = df[df['Human_Polarity'] == polarity]
        fig.add_trace(go.Scatter(
            x=subset_gt['UMAP_X'], y=subset_gt['UMAP_Y'],
            mode='markers',
            marker=dict(
                color=neon_colors[polarity],
                size=6.5, opacity=0.85,
                line=dict(width=0.5, color='#FFFFFF') 
            ),
            name=f"{polarity} (Human)",
            legendgroup=polarity,
            showlegend=True,
            hovertext=subset_gt.apply(lambda row: build_hover_text(row, 'Human_Polarity'), axis=1),
            hoverinfo="text"
        ), row=1, col=1)

        # --- RIGHT PANEL: Colored by what the MACHINE calculated ---
        subset_machine = df[df['Machine_Polarity'] == polarity]
        fig.add_trace(go.Scatter(
            x=subset_machine['UMAP_X'], y=subset_machine['UMAP_Y'],
            mode='markers',
            marker=dict(
                color=neon_colors[polarity],
                size=6.5, opacity=0.85,
                line=dict(width=0.5, color='#FFFFFF')
            ),
            name=f"{polarity} (BERT)",
            legendgroup=polarity,
            showlegend=False, 
            hovertext=subset_machine.apply(lambda row: build_hover_text(row, 'Machine_Polarity'), axis=1),
            hoverinfo="text"
        ), row=1, col=2)

    # Layout Aesthetics - Matrix Style Title
    fig.update_layout(
        title={
            'text': "<span style='color:#39FF14; font-family: \"Courier New\", Courier, monospace; font-size:24px; text-shadow: 0px 0px 8px #39FF14;'>LATENT SPACE SEMANTIC ANALYSIS</span><br>"
                    "<span style='font-size:13px; color:#A0A0A0; font-family: Helvetica, Arial, sans-serif;'>Comparing Human Product Evaluation vs. Machine Emotion Transfer (Zero-Shot 28->2)</span>",
            'x': 0.5, 'xanchor': 'center'
        },
        template='plotly_dark',
        plot_bgcolor='#050505', paper_bgcolor='#050505', # Ultra dark background
        font=dict(family="Helvetica Neue, Arial, sans-serif", color="#E0E0E0"),
        legend=dict(
            title="<span style='color:#39FF14'><b>Sentiment Polarity</b></span>",
            itemsizing='constant',
            font=dict(size=12),
            y=1, x=1.02
        ),
        margin=dict(l=40, r=40, t=110, b=240), 
        height=800, 
        # Deep black hover background, thin neon green border
        hoverlabel=dict(bgcolor="#000000", font_size=13, font_family="Courier New", bordercolor="#39FF14")
    )

    # High-Tech Radar Axis Configuration
    axis_config = dict(
        showgrid=True, gridwidth=1, gridcolor='rgba(57, 255, 20, 0.15)', # Faint matrix green grid
        zeroline=True, zerolinewidth=2, zerolinecolor='rgba(57, 255, 20, 0.5)', # Brighter radar crosshair
        showline=True, linewidth=2, linecolor='#39FF14', # Solid Matrix green axis borders
        mirror=True, # Creates a full bounding box around each plot
        showticklabels=False
    )
    
    fig.update_xaxes(title_text="<span style='color:#A0A0A0'>Semantic Axis X (Contextual Proximity)</span>", **axis_config, row=1, col=1)
    fig.update_xaxes(title_text="<span style='color:#A0A0A0'>Semantic Axis X (Contextual Proximity)</span>", **axis_config, row=1, col=2)
    fig.update_yaxes(title_text="<span style='color:#A0A0A0'>Semantic Axis Y</span>", **axis_config, row=1, col=1)
    fig.update_yaxes(title_text="<span style='color:#A0A0A0'>Semantic Axis Y</span>", **axis_config, row=1, col=2)

    # Analytical Text Box anchored compactly below the plot
    methodology_text = (
        "<b>Analytical Conclusion:</b><br>"
        "These scatter plots visualize 768-dimensional BERT embeddings reduced to 2D via UMAP. Points are colored by Polarity.<br>"
        "<b>Left:</b> Ground Truth as annotated by humans based on product evaluation rules (IMDB).<br>"
        "<b>Right:</b> Machine prediction derived by collapsing 28 specific emotional states into a binary heuristic.<br><br>"
        "<span style='color:#A0A0A0;'><i>Hovering over discrepancies reveals the necessity of human annotation. For example, a machine may interpret<br>"
        "the emotion 'Fear' as strictly Negative, while a human recognizes it as a Positive reaction to a horror movie.</i></span>"
    )

    fig.add_annotation(
        text=methodology_text,
        xref="paper", yref="paper",
        x=0.5, y=-0.16,      
        yanchor="top",       
        showarrow=False,
        font=dict(size=12, color="#CCCCCC"),
        align="left",
        bordercolor="#333333", borderwidth=1, borderpad=12,
        bgcolor="#000000"
    )

    output_path = "data/processed/umap_matrix_comparative.html"
    fig.write_html(output_path)
    print(f"✅ Professional Matrix-Style Dashboard saved to: {output_path}")
    fig.show()

if __name__ == "__main__":
    print("Loading mapped IMDB dataset...")
    imdb_df = pd.read_csv("data/processed/imdb_mapped.csv")

    subset_df = imdb_df.head(1500).copy()
    texts = subset_df['text'].tolist()

    embeddings = extract_embeddings(texts, batch_size=32)
    create_neon_comparative_umap(embeddings, subset_df)