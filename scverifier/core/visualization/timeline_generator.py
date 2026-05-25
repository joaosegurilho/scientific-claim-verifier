"""Timeline visualization for proposition similarity across years."""

from typing import List, Tuple, Dict, Any
from collections import defaultdict
import plotly.graph_objects as go
from scverifier.data.models import Proposition, Paper


class TimelineGenerator:
    """Generates timeline visualizations for proposition similarity."""

    def __init__(self):
        pass

    def get_all_propositions_by_year(self, papers: Dict[str, Paper]) -> Dict[int, int]:
        """Get total proposition count for each year across all papers.

        Args:
            papers: Dictionary of paper_id -> Paper objects

        Returns:
            Dictionary mapping year -> total proposition count
        """
        year_counts = defaultdict(int)

        for paper in papers.values():
            if paper.year and paper.propositions:
                year_counts[paper.year] += len(paper.propositions)

        return dict(year_counts)

    def aggregate_by_year(
        self, propositions_with_scores: List[Tuple[Proposition, float]], papers: Dict[str, Any]
    ) -> Dict[int, Dict[str, Any]]:
        """Aggregate propositions by year with average similarity.

        Args:
            propositions_with_scores: List of (Proposition, similarity_score) tuples
            papers: Dictionary of paper_id -> Paper objects

        Returns:
            Dictionary mapping year -> {
                'count': int,
                'avg_similarity': float,
                'propositions': List[Tuple[Proposition, float]]
            }
        """
        year_data = defaultdict(lambda: {"count": 0, "total_similarity": 0.0, "propositions": []})

        for prop, score in propositions_with_scores:
            paper = papers.get(prop.paper_id)
            if paper and paper.year:
                year = paper.year
                year_data[year]["count"] += 1
                year_data[year]["total_similarity"] += score
                year_data[year]["propositions"].append((prop, score))

        # Calculate averages
        result = {}
        for year, data in year_data.items():
            result[year] = {
                "count": data["count"],
                "avg_similarity": data["total_similarity"] / data["count"],
                "propositions": data["propositions"],
            }

        return result

    def similarity_to_color(self, similarity: float, min_sim: float, max_sim: float) -> str:
        """Convert similarity score to RGB color string using absolute 0-1 gradient.

        Red (0.0) -> Yellow (0.5) -> Green (1.0)

        Args:
            similarity: Similarity score (0.0 to 1.0)
            min_sim: Minimum similarity in the dataset (unused, kept for compatibility)
            max_sim: Maximum similarity in the dataset (unused, kept for compatibility)

        Returns:
            RGB color string like 'rgb(40, 167, 69)'
        """
        # Clamp similarity to 0-1 range
        norm = max(0.0, min(1.0, similarity))

        # Color gradient based on absolute similarity value:
        # 0.0: Red rgb(220, 53, 69)
        # 0.5: Yellow rgb(255, 193, 7)
        # 1.0: Green rgb(40, 167, 69)

        if norm >= 0.5:
            # Yellow to Green zone (0.5 to 1.0)
            ratio = (norm - 0.5) * 2  # 0 to 1
            red = int(255 - (255 - 40) * ratio)
            green = int(193 + (167 - 193) * ratio)
            blue = int(7 + (69 - 7) * ratio)
        else:
            # Red to Yellow zone (0.0 to 0.5)
            ratio = norm * 2  # 0 to 1
            red = int(220 + (255 - 220) * ratio)
            green = int(53 + (193 - 53) * ratio)
            blue = int(69 + (7 - 69) * ratio)

        return f"rgb({red}, {green}, {blue})"

    def generate_bar_chart(self, year_data: Dict[int, Dict[str, Any]], query: str,
                          total_props_by_year: Dict[int, int] = None) -> str:
        """Generate interactive Plotly bar chart HTML with dual y-axes for better visibility.

        Uses dual y-axes:
        - Left axis: Relevant propositions (colored by similarity)
        - Right axis: Total propositions (background, gray)
        This allows both to scale independently for better visibility.

        Args:
            year_data: Aggregated data by year
            query: Original user query
            total_props_by_year: Total proposition counts per year (optional)

        Returns:
            HTML string containing the Plotly chart
        """
        if not year_data:
            return "<p>No data available for visualization.</p>"

        # Sort by year
        years = sorted(year_data.keys())
        counts = [year_data[year]["count"] for year in years]
        avg_similarities = [year_data[year]["avg_similarity"] for year in years]

        # Generate colors based on absolute similarity (0-1 scale)
        colors = [self.similarity_to_color(sim, 0.0, 1.0) for sim in avg_similarities]

        # Create hover text with details
        hover_text = [
            f"Year: {year}<br>"
            f"Relevant Propositions: {year_data[year]['count']}<br>"
            f"Avg Similarity: {year_data[year]['avg_similarity']:.3f}"
            for year in years
        ]

        # Build traces
        data = []

        # Background bars for total propositions (if provided) - on secondary y-axis
        if total_props_by_year:
            total_counts = [total_props_by_year.get(year, 0) for year in years]
            data.append(
                go.Bar(
                    x=years,
                    y=total_counts,
                    name='Total Propositions',
                    marker=dict(
                        color='rgba(200,200,200,0.3)',
                        line=dict(color='rgba(150,150,150,0.5)', width=1)
                    ),
                    hovertext=[f"Year: {year}<br>Total Propositions: {total_props_by_year.get(year, 0)}"
                              for year in years],
                    hoverinfo='text',
                    yaxis='y2'  # Use secondary y-axis
                )
            )

        # Foreground bars for relevant propositions - on primary y-axis
        data.append(
            go.Bar(
                x=years,
                y=counts,
                name='Relevant Propositions',
                marker=dict(color=colors, line=dict(color="rgba(0,0,0,0.2)", width=1)),
                hovertext=hover_text,
                hoverinfo="text",
                yaxis='y1'  # Use primary y-axis
            )
        )

        # Create bar chart
        fig = go.Figure(data=data)

        # Update layout with dual y-axes
        fig.update_layout(
            title=dict(text=f"Proposition Timeline for: '{query}'", x=0.5, xanchor="center", font=dict(size=18)),
            xaxis=dict(
                title="Year of Publication", tickmode="linear", tick0=min(years), dtick=1 if len(years) <= 20 else 2
            ),
            yaxis=dict(
                title="Relevant Propositions",
                # titlefont=dict(color="rgb(55, 126, 184)"),
                tickfont=dict(color="rgb(55, 126, 184)")
            ),
            yaxis2=dict(
                title="Total Propositions",
                # titlefont=dict(color="rgb(150, 150, 150)"),
                tickfont=dict(color="rgb(150, 150, 150)"),
                overlaying='y',
                side='right'
            ),
            hovermode="closest",
            plot_bgcolor="rgba(240,240,240,0.5)",
            height=500,
            margin=dict(l=60, r=220, t=80, b=80),
            barmode='overlay',
            showlegend=True,
            legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)")
        )

        # Add color scale legend on the right side
        fig.add_annotation(
            text="<b>Similarity Scale:</b><br>"
            "<span style='color:rgb(40,167,69)'>■</span> 1.0 (High)<br>"
            "<span style='color:rgb(147,180,69)'>■</span> 0.75<br>"
            "<span style='color:rgb(255,193,7)'>■</span> 0.5 (Medium)<br>"
            "<span style='color:rgb(237,123,38)'>■</span> 0.25<br>"
            "<span style='color:rgb(220,53,69)'>■</span> 0.0 (Low)",
            xref="paper",
            yref="paper",
            x=1.10,
            y=0.5,
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font=dict(size=10),
            align="left",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
            borderpad=8,
        )

        # Convert to HTML
        return fig.to_html(full_html=False, include_plotlyjs="cdn")

    def generate_cumulative_chart(self, year_data: Dict[int, Dict[str, Any]], query: str) -> str:
        """Generate cumulative evidence chart showing proposition accumulation over time.

        Args:
            year_data: Aggregated data by year
            query: Original user query

        Returns:
            HTML string containing the Plotly cumulative chart
        """
        if not year_data:
            return "<p>No data available for visualization.</p>"

        # Sort by year and calculate cumulative counts
        years = sorted(year_data.keys())
        cumulative_counts = []
        total = 0

        for year in years:
            total += year_data[year]["count"]
            cumulative_counts.append(total)

        # Create line chart
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=years,
            y=cumulative_counts,
            mode='lines+markers',
            name='Cumulative Propositions',
            line=dict(color='rgb(55, 126, 184)', width=3),
            marker=dict(size=8, color='rgb(55, 126, 184)'),
            fill='tozeroy',
            fillcolor='rgba(55, 126, 184, 0.2)',
            hovertemplate='Year: %{x}<br>Cumulative: %{y}<extra></extra>'
        ))

        fig.update_layout(
            title=dict(text=f"Cumulative Evidence Over Time for: '{query}'", x=0.5, xanchor="center", font=dict(size=18)),
            xaxis=dict(
                title="Year of Publication",
                tickmode="linear",
                tick0=min(years),
                dtick=1 if len(years) <= 20 else 2
            ),
            yaxis=dict(title="Cumulative Number of Propositions"),
            hovermode="closest",
            plot_bgcolor="rgba(240,240,240,0.5)",
            height=500,
            margin=dict(l=60, r=60, t=80, b=80),
        )

        return fig.to_html(full_html=False, include_plotlyjs="cdn")

    def generate_heatmap(self, year_data: Dict[int, Dict[str, Any]], papers: Dict[str, Any], query: str, top_n: int = 20) -> str:
        """Generate horizontal bar chart showing top papers by similarity.
        
        Each paper appears as a horizontal bar where:
        - Y-axis: Paper title and year
        - X-axis: Average similarity score
        - Color: Similarity score (red to green gradient)
        - Bar length: Similarity value

        Args:
            year_data: Aggregated data by year
            papers: Dictionary of paper_id -> Paper objects
            query: Original user query
            top_n: Number of top papers to display

        Returns:
            HTML string containing the Plotly bar chart
        """
        if not year_data:
            return "<p>No data available for visualization.</p>"

        # Calculate stats for each paper
        paper_info = {}  # paper_id -> {year, avg_similarity, prop_count, title}
        
        for year, data in year_data.items():
            for prop, score in data["propositions"]:
                paper_id = prop.paper_id
                
                if paper_id not in paper_info:
                    paper = papers.get(paper_id)
                    if paper:
                        paper_info[paper_id] = {
                            "year": paper.year,
                            "title": paper.title,
                            "total_similarity": 0.0,
                            "prop_count": 0
                        }
                
                if paper_id in paper_info:
                    paper_info[paper_id]["total_similarity"] += score
                    paper_info[paper_id]["prop_count"] += 1

        # Calculate average similarity for each paper
        for paper_id in paper_info:
            count = paper_info[paper_id]["prop_count"]
            paper_info[paper_id]["avg_similarity"] = paper_info[paper_id]["total_similarity"] / count

        # Get top N papers by average similarity
        sorted_papers = sorted(
            paper_info.items(), 
            key=lambda x: x[1]["avg_similarity"], 
            reverse=True
        )[:top_n]

        if not sorted_papers:
            return "<p>No data available for visualization.</p>"

        # Prepare horizontal bar chart data
        paper_labels = []
        similarities = []
        colors = []
        hover_texts = []

        for paper_id, info in sorted_papers:
            # Truncate title for y-axis label
            title = info["title"][:60] + "..." if len(info["title"]) > 60 else info["title"]
            paper_labels.append(f"{title}<br>({info['year']}) - {info['prop_count']} props")
            
            similarities.append(info["avg_similarity"])
            colors.append(self.similarity_to_color(info["avg_similarity"], 0.0, 1.0))
            
            # Full title for hover
            full_title = info["title"]
            hover_texts.append(
                f"<b>{full_title}</b><br>"
                f"Year: {info['year']}<br>"
                f"Avg Similarity: {info['avg_similarity']:.3f}<br>"
                f"Propositions: {info['prop_count']}"
            )

        # Create horizontal bar chart (reverse so highest is on top)
        fig = go.Figure(data=go.Bar(
            x=similarities,
            y=paper_labels,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='rgba(0,0,0,0.2)', width=1)
            ),
            hovertext=hover_texts,
            hoverinfo='text'
        ))

        fig.update_layout(
            title=dict(
                text=f"Top {len(sorted_papers)} Papers by Similarity for: '{query}'", 
                x=0.5, 
                xanchor="center", 
                font=dict(size=18)
            ),
            xaxis=dict(
                title="Average Similarity Score",
                range=[0, 1],
                gridcolor='rgba(200,200,200,0.3)'
            ),
            yaxis=dict(
                title="",
                autorange="reversed"  # Highest similarity at top
            ),
            plot_bgcolor="rgba(240,240,240,0.5)",
            height=max(500, len(sorted_papers) * 45),  # Increased spacing between bars
            margin=dict(l=350, r=80, t=100, b=80),
            bargap=0.3,  # Add gap between bars for better separation
            hovermode='closest',
            showlegend=False
        )

        return fig.to_html(full_html=False, include_plotlyjs="cdn")

    def generate_visualization_data(
        self, query: str, propositions_with_scores: List[Tuple[Proposition, float]],
        papers: Dict[str, Any], total_props_by_year: Dict[int, int] = None
    ) -> Dict[str, Any]:
        """Generate complete visualization data including all charts and summary.

        Args:
            query: User query
            propositions_with_scores: List of (Proposition, similarity_score) tuples
            papers: Dictionary of paper_id -> Paper objects
            total_props_by_year: Total proposition counts per year (optional)

        Returns:
            Dictionary with all visualization HTMLs, 'year_data', and 'summary'
        """
        # Aggregate by year
        year_data = self.aggregate_by_year(propositions_with_scores, papers)

        # Generate all visualizations
        timeline_html = self.generate_bar_chart(year_data, query, total_props_by_year)
        cumulative_html = self.generate_cumulative_chart(year_data, query)
        heatmap_html = self.generate_heatmap(year_data, papers, query)

        # Generate summary statistics
        total_propositions = len(propositions_with_scores)
        years_covered = sorted(year_data.keys())
        year_range = f"{min(years_covered)}-{max(years_covered)}" if years_covered else "N/A"
        avg_overall_similarity = (
            sum(data["avg_similarity"] * data["count"] for data in year_data.values()) / total_propositions
            if total_propositions > 0
            else 0.0
        )

        summary = {
            "total_propositions": total_propositions,
            "years_covered": len(years_covered),
            "year_range": year_range,
            "avg_similarity": avg_overall_similarity,
            "peak_year": max(year_data.items(), key=lambda x: x[1]["count"])[0] if year_data else None,
            "most_similar_year": max(year_data.items(), key=lambda x: x[1]["avg_similarity"])[0] if year_data else None,
        }

        return {
            "timeline_html": timeline_html,
            "cumulative_html": cumulative_html,
            "heatmap_html": heatmap_html,
            "year_data": year_data,
            "summary": summary
        }
