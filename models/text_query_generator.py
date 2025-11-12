"""
Text-Conditioned Query Generator for AdaPoinTr
Generates queries conditioned on both encoder features and text embeddings
"""

import torch
import torch.nn as nn


class TextConditionedQueryGenerator(nn.Module):
    """
    Text-Conditioned Query Generator that creates queries based on both
    geometric features from the encoder and semantic features from text.

    This generator concatenates pooled encoder features with text embeddings
    and uses an MLP to generate dynamic queries.

    Args:
        encoder_dim: int, dimension of encoder output features (e.g., 384)
        text_dim: int, dimension of text features from CLIP (768 for CLIP-Large)
        num_queries: int, number of queries to generate (e.g., 256 or 512)
        query_dim: int, dimension of each query feature (e.g., 384)
    """

    def __init__(self, encoder_dim, text_dim, num_queries, query_dim):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.text_dim = text_dim
        self.num_queries = num_queries
        self.query_dim = query_dim

        # Pool encoder features to fixed dimension
        self.encoder_pool = nn.AdaptiveMaxPool1d(1)

        # MLP for query generation
        # Input: [encoder_dim + text_dim] -> Output: [num_queries * query_dim]
        self.mlp = nn.Sequential(
            nn.Linear(encoder_dim + text_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, num_queries * query_dim)
        )

        print(f'[TEXT_QUERY_GENERATOR] Initialized with:')
        print(f'  - Encoder dim: {encoder_dim}')
        print(f'  - Text dim: {text_dim}')
        print(f'  - Num queries: {num_queries}')
        print(f'  - Query dim: {query_dim}')

    def forward(self, encoder_output, text_pooled):
        """
        Generate queries conditioned on encoder features and text embeddings.

        Args:
            encoder_output: torch.Tensor, [B, N, encoder_dim] - encoder features
            text_pooled: torch.Tensor, [B, text_dim] - pooled text features

        Returns:
            queries: torch.Tensor, [B, num_queries, query_dim] - generated queries
        """
        B = encoder_output.size(0)

        # Pool encoder output: [B, encoder_dim, N] -> [B, encoder_dim, 1] -> [B, encoder_dim]
        encoder_pooled = self.encoder_pool(encoder_output.transpose(1, 2)).squeeze(-1)  # [B, encoder_dim]

        # Concatenate pooled encoder features with text features
        combined_features = torch.cat([encoder_pooled, text_pooled], dim=1)  # [B, encoder_dim + text_dim]

        # Generate queries using MLP
        query_flat = self.mlp(combined_features)  # [B, num_queries * query_dim]

        # Reshape to [B, num_queries, query_dim]
        queries = query_flat.view(B, self.num_queries, self.query_dim)

        return queries


class AdaptiveTextQueryGenerator(nn.Module):
    """
    Adaptive Text-Conditioned Query Generator for AdaPoinTr.

    This is a more advanced version that generates both input queries (QI) and
    output queries (QO), similar to the adaptive query generation in AdaPoinTr paper.

    Args:
        encoder_dim: int, dimension of encoder output features
        text_dim: int, dimension of text features from CLIP
        input_proxy_dim: int, dimension of input point proxy features
        num_queries_input: int, number of queries from input (e.g., 256)
        num_queries_output: int, number of queries for missing parts (e.g., 256)
        query_dim: int, dimension of each query feature
    """

    def __init__(self, encoder_dim, text_dim, input_proxy_dim,
                 num_queries_input, num_queries_output, query_dim):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.text_dim = text_dim
        self.input_proxy_dim = input_proxy_dim
        self.num_queries_input = num_queries_input
        self.num_queries_output = num_queries_output
        self.query_dim = query_dim

        # Pool encoder and input proxy features
        self.encoder_pool = nn.AdaptiveMaxPool1d(1)
        self.input_proxy_pool = nn.AdaptiveMaxPool1d(1)

        # Coordinate projection for input queries (from input point proxies)
        self.coord_proj_input = nn.Linear(input_proxy_dim, num_queries_input * 3)

        # Coordinate projection for output queries (conditioned on encoder + text)
        self.coord_proj_output = nn.Linear(encoder_dim + text_dim, num_queries_output * 3)

        # Query feature generation MLPs
        self.query_mlp_input = nn.Sequential(
            nn.Linear(input_proxy_dim + 3, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, query_dim)
        )

        self.query_mlp_output = nn.Sequential(
            nn.Linear(encoder_dim + text_dim + 3, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, query_dim)
        )

        # Query selection/ranking module (optional)
        self.query_ranking = nn.Sequential(
            nn.Linear(3, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        print(f'[ADAPTIVE_TEXT_QUERY_GENERATOR] Initialized with:')
        print(f'  - Encoder dim: {encoder_dim}, Text dim: {text_dim}')
        print(f'  - Input proxy dim: {input_proxy_dim}')
        print(f'  - Num input queries: {num_queries_input}')
        print(f'  - Num output queries: {num_queries_output}')
        print(f'  - Query dim: {query_dim}')

    def forward(self, encoder_output, text_pooled, input_proxies, num_selected=None):
        """
        Generate adaptive queries conditioned on encoder, text, and input proxies.

        Args:
            encoder_output: torch.Tensor, [B, N, encoder_dim]
            text_pooled: torch.Tensor, [B, text_dim]
            input_proxies: torch.Tensor, [B, M, input_proxy_dim]
            num_selected: int or None, number of queries to select (if None, use all)

        Returns:
            queries: torch.Tensor, [B, num_selected or (num_queries_input + num_queries_output), query_dim]
            coordinates: torch.Tensor, [B, num_selected or (num_queries_input + num_queries_output), 3]
        """
        B = encoder_output.size(0)

        # Pool features
        encoder_pooled = self.encoder_pool(encoder_output.transpose(1, 2)).squeeze(-1)  # [B, encoder_dim]
        input_proxy_pooled = self.input_proxy_pool(input_proxies.transpose(1, 2)).squeeze(-1)  # [B, input_proxy_dim]

        # Generate coordinates for input queries
        coords_input_flat = self.coord_proj_input(input_proxy_pooled)  # [B, num_queries_input * 3]
        coords_input = coords_input_flat.view(B, self.num_queries_input, 3)  # [B, num_queries_input, 3]

        # Generate coordinates for output queries (conditioned on encoder + text)
        encoder_text_combined = torch.cat([encoder_pooled, text_pooled], dim=1)  # [B, encoder_dim + text_dim]
        coords_output_flat = self.coord_proj_output(encoder_text_combined)  # [B, num_queries_output * 3]
        coords_output = coords_output_flat.view(B, self.num_queries_output, 3)  # [B, num_queries_output, 3]

        # Generate query features for input queries
        # Expand pooled features and concatenate with coordinates
        input_proxy_pooled_expanded = input_proxy_pooled.unsqueeze(1).expand(-1, self.num_queries_input, -1)  # [B, M_I, input_proxy_dim]
        input_query_input = torch.cat([input_proxy_pooled_expanded, coords_input], dim=-1)  # [B, M_I, input_proxy_dim + 3]
        queries_input = self.query_mlp_input(input_query_input)  # [B, M_I, query_dim]

        # Generate query features for output queries
        encoder_text_expanded = encoder_text_combined.unsqueeze(1).expand(-1, self.num_queries_output, -1)  # [B, M_O, encoder_dim + text_dim]
        output_query_input = torch.cat([encoder_text_expanded, coords_output], dim=-1)  # [B, M_O, encoder_dim + text_dim + 3]
        queries_output = self.query_mlp_output(output_query_input)  # [B, M_O, query_dim]

        # Concatenate input and output queries
        all_queries = torch.cat([queries_input, queries_output], dim=1)  # [B, M_I + M_O, query_dim]
        all_coords = torch.cat([coords_input, coords_output], dim=1)  # [B, M_I + M_O, 3]

        # Optional: Query selection based on ranking
        if num_selected is not None and num_selected < (self.num_queries_input + self.num_queries_output):
            # Rank queries based on coordinates
            query_scores = self.query_ranking(all_coords)  # [B, M_I + M_O, 1]
            query_scores = query_scores.squeeze(-1)  # [B, M_I + M_O]

            # Select top queries
            _, idx = torch.topk(query_scores, num_selected, dim=1)  # [B, num_selected]
            idx_expanded = idx.unsqueeze(-1).expand(-1, -1, self.query_dim)  # [B, num_selected, query_dim]
            selected_queries = torch.gather(all_queries, 1, idx_expanded)  # [B, num_selected, query_dim]

            idx_coords = idx.unsqueeze(-1).expand(-1, -1, 3)  # [B, num_selected, 3]
            selected_coords = torch.gather(all_coords, 1, idx_coords)  # [B, num_selected, 3]

            return selected_queries, selected_coords
        else:
            return all_queries, all_coords


if __name__ == '__main__':
    # Test the text-conditioned query generator
    print("Testing TextConditionedQueryGenerator...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Parameters
    B = 4  # batch size
    N = 256  # number of encoder points
    encoder_dim = 384
    text_dim = 768
    num_queries = 256
    query_dim = 384

    # Create generator
    generator = TextConditionedQueryGenerator(
        encoder_dim=encoder_dim,
        text_dim=text_dim,
        num_queries=num_queries,
        query_dim=query_dim
    ).to(device)

    # Create dummy inputs
    encoder_output = torch.randn(B, N, encoder_dim).to(device)
    text_pooled = torch.randn(B, text_dim).to(device)

    # Generate queries
    print(f"\nGenerating queries...")
    queries = generator(encoder_output, text_pooled)

    print(f"Queries shape: {queries.shape}")  # Should be [4, 256, 384]
    assert queries.shape == (B, num_queries, query_dim), "Query shape mismatch!"

    print("\nTest passed!")

    # Test adaptive query generator
    print("\n" + "="*50)
    print("Testing AdaptiveTextQueryGenerator...")

    input_proxy_dim = 128
    num_queries_input = 128
    num_queries_output = 128

    adaptive_generator = AdaptiveTextQueryGenerator(
        encoder_dim=encoder_dim,
        text_dim=text_dim,
        input_proxy_dim=input_proxy_dim,
        num_queries_input=num_queries_input,
        num_queries_output=num_queries_output,
        query_dim=query_dim
    ).to(device)

    # Create dummy inputs
    input_proxies = torch.randn(B, N, input_proxy_dim).to(device)

    # Generate queries
    print(f"\nGenerating adaptive queries...")
    queries, coords = adaptive_generator(encoder_output, text_pooled, input_proxies)

    print(f"Queries shape: {queries.shape}")  # Should be [4, 256, 384]
    print(f"Coordinates shape: {coords.shape}")  # Should be [4, 256, 3]

    # Test with selection
    num_selected = 200
    queries_selected, coords_selected = adaptive_generator(
        encoder_output, text_pooled, input_proxies, num_selected=num_selected
    )

    print(f"\nWith selection (top {num_selected}):")
    print(f"Selected queries shape: {queries_selected.shape}")  # Should be [4, 200, 384]
    print(f"Selected coordinates shape: {coords_selected.shape}")  # Should be [4, 200, 3]

    print("\nAll tests passed!")
