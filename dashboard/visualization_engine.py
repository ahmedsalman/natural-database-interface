"""
Visualization Engine - Chart Generation with Plotly
Implements approved chart type selection and rendering logic
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

from dashboard.models.kpi_models import KPI

logger = logging.getLogger(__name__)


class VisualizationEngine:
    """
    Generates Plotly charts based on KPI data
    
    Chart selection rules (APPROVED):
    - Time series: Always line chart (Q4A: Option A)
    - Categorical < 5: Pie chart
    - Categorical 5-15: Vertical bar chart
    - Categorical > 15: Horizontal bar chart (top 20)
    - Single metric: Big number card with optional sparkline
    """
    
    # Chart configuration
    DEFAULT_HEIGHT = 400
    METRIC_CARD_HEIGHT = 200
    TALL_CHART_HEIGHT = 600  # For horizontal bar with many items
    
    # Color schemes
    COLOR_SCHEME = px.colors.qualitative.Set3
    PRIMARY_COLOR = '#1f77b4'
    SUCCESS_COLOR = '#2ca02c'
    WARNING_COLOR = '#ff7f0e'
    DANGER_COLOR = '#d62728'
    
    def __init__(self, theme: str = 'plotly_white'):
        """
        Initialize visualization engine
        
        Args:
            theme: Plotly template theme
        """
        self.theme = theme
        logger.info(f"Visualization engine initialized with theme: {theme}")
    
    def generate_chart(
        self,
        kpi: KPI,
        data: pd.DataFrame,
        time_range_days: Optional[int] = None
    ) -> go.Figure:
        """
        Generate appropriate chart for KPI
        
        Args:
            kpi: KPI definition
            data: Query result data
            time_range_days: Time range used (for context)
            
        Returns:
            Plotly Figure object
        """
        if data.empty:
            return self._create_empty_chart(kpi.name)
        
        # Route to appropriate chart type
        if kpi.chart_type == 'metric':
            return self._create_metric_card(kpi, data, time_range_days)
        
        elif kpi.chart_type == 'line':
            return self._create_line_chart(kpi, data)
        
        elif kpi.chart_type == 'bar':
            return self._create_bar_chart(kpi, data)
        
        elif kpi.chart_type == 'pie':
            return self._create_pie_chart(kpi, data)
        
        elif kpi.chart_type == 'table':
            return self._create_table(kpi, data)
        
        else:
            logger.warning(f"Unknown chart type: {kpi.chart_type}")
            return self._create_empty_chart(kpi.name)
    
    def _create_metric_card(
        self,
        kpi: KPI,
        data: pd.DataFrame,
        time_range_days: Optional[int]
    ) -> go.Figure:
        """
        Create big number card with optional sparkline and delta
        
        Expected data format:
        - 'value': Current value (required)
        - 'previous_value': Previous period value (optional, for delta)
        - 'trend_data': List of recent values (optional, for sparkline)
        """
        # Extract current value
        if 'value' not in data.columns:
            # Step 1: get all columns
            all_cols = data.columns.tolist()

            # Step 2: dynamically detect numeric columns
            numeric_cols = []
            for col in all_cols:
                # Try converting to numeric
                try:
                    pd.to_numeric(data[col])
                    numeric_cols.append(col)
                except Exception:
                    continue

            # Step 3: pick the first numeric column if exists
            if numeric_cols:
                value_col = numeric_cols[0]
                current_value = pd.to_numeric(data[value_col]).iloc[0]
            else:
                value_col = None
                current_value = None

            if len(numeric_cols) > 0:
                current_value = data[numeric_cols[0]].iloc[0]
            else:
                return self._create_empty_chart(kpi.name)
        else:
            current_value = data['value'].iloc[0]
        
        # Extract previous value for delta (if available)
        previous_value = None
        if 'previous_value' in data.columns:
            previous_value = data['previous_value'].iloc[0]
        
        # Extract trend data for sparkline (if available)
        trend_data = None
        if 'trend_data' in data.columns and data['trend_data'].iloc[0] is not None:
            trend_data = data['trend_data'].iloc[0]
        
        # Create figure
        fig = go.Figure()
        
        # Main indicator
        mode = "number"
        if previous_value is not None and pd.notna(previous_value):
            mode = "number+delta"
        
        fig.add_trace(go.Indicator(
            mode=mode,
            value=current_value,
            delta={
                'reference': previous_value,
                'relative': True,
                'valueformat': '.1%',
                'increasing': {'color': self.SUCCESS_COLOR},
                'decreasing': {'color': self.DANGER_COLOR}
            } if previous_value else None,
            title={
                'text': kpi.name,
                'font': {'size': 20}
            },
            number={
                'valueformat': self._get_number_format(current_value),
                'font': {'size': 48}
            },
            domain={'x': [0, 1], 'y': [0.4, 1]}
        ))
        
        # Add sparkline if trend data available
        if trend_data and len(trend_data) > 1:
            fig.add_trace(go.Scatter(
                y=trend_data,
                mode='lines',
                line={
                    'color': self.PRIMARY_COLOR,
                    'width': 2
                },
                fill='tozeroy',
                fillcolor=f'rgba(31, 119, 180, 0.2)',
                showlegend=False,
                hovertemplate='%{y:,.0f}<extra></extra>',
                xaxis='x2',
                yaxis='y2'
            ))
        
        # Layout
        fig.update_layout(
            height=self.METRIC_CARD_HEIGHT,
            template=self.theme,
            margin=dict(l=20, r=20, t=60, b=20),
            xaxis2={
                'domain': [0.1, 0.9],
                'anchor': 'y2',
                'showgrid': False,
                'showticklabels': False,
                'zeroline': False
            },
            yaxis2={
                'domain': [0, 0.35],
                'anchor': 'x2',
                'showgrid': False,
                'showticklabels': False,
                'zeroline': False
            }
        )
        
        # Add time range context
        if time_range_days:
            fig.add_annotation(
                text=f"Last {time_range_days} days",
                xref="paper", yref="paper",
                x=0.5, y=-0.1,
                showarrow=False,
                font=dict(size=12, color='gray')
            )
        
        return fig
    
    def _create_line_chart(self, kpi: KPI, data: pd.DataFrame) -> go.Figure:
        """
        Create time series line chart (ALWAYS for time series - Q4A)
        
        Expected data format:
        - Date column (datetime type)
        - One or more numeric columns
        """
        # Detect date column
        date_cols = data.select_dtypes(include=['datetime64', 'datetime']).columns
        
        if len(date_cols) == 0:
            # Try to parse string columns as dates
            for col in data.columns:
                try:
                    data[col] = pd.to_datetime(data[col])
                    date_cols = [col]
                    break
                except:
                    continue
        
        if len(date_cols) == 0:
            logger.warning(f"No date column found for line chart: {kpi.name}")
            return self._create_empty_chart(kpi.name)
        
        date_col = date_cols[0]
        
        # Get numeric columns
        numeric_cols = data.select_dtypes(include='number').columns.tolist()
        
        if len(numeric_cols) == 0:
            return self._create_empty_chart(kpi.name)
        
        # Create figure
        fig = go.Figure()
        
        # Add line for each numeric column
        for col in numeric_cols:
            fig.add_trace(go.Scatter(
                x=data[date_col],
                y=data[col],
                mode='lines+markers',
                name=col if len(numeric_cols) > 1 else kpi.name,
                line={'width': 2},
                marker={'size': 6},
                hovertemplate='%{y:,.0f}<extra></extra>'
            ))
        
        # Layout
        fig.update_layout(
            title=kpi.name,
            xaxis_title='Date',
            yaxis_title='Value',
            height=self.DEFAULT_HEIGHT,
            template=self.theme,
            hovermode='x unified',
            showlegend=len(numeric_cols) > 1
        )
        
        # Format y-axis
        fig.update_yaxes(
            tickformat=self._get_number_format(data[numeric_cols[0]].iloc[0])
        )
        
        return fig
    
    def _create_bar_chart(self, kpi: KPI, data: pd.DataFrame) -> go.Figure:
        """
        Create bar chart with dynamic orientation (Q4B rules)
        
        Rules (APPROVED):
        - < 5 categories: Use pie chart instead (warn user)
        - 5-15 categories: Vertical bar chart
        - > 15 categories: Horizontal bar chart, top 20, sorted descending
        """
        # Step 1: get column candidates
        category_col = data.select_dtypes(include='object').columns.tolist()
        value_col = data.select_dtypes(include='number').columns.tolist()

        # Step 2: coerce numeric-like object columns
        for col in category_col:
            converted = pd.to_numeric(data[col], errors='coerce')
            if converted.notna().any():
                data[col] = converted
                value_col.append(col)

        # Step 3: re-evaluate after coercion
        category_col = data.select_dtypes(include='object').columns.tolist()
        value_col = data.select_dtypes(include='number').columns.tolist()
        # Identify category and value columns
        category_col = data.select_dtypes(include='object').columns[0]
        value_col = data.select_dtypes(include='number').columns[0]

        num_categories = len(data)
        
        # < 5 categories: Suggest pie chart
        if num_categories < 5:
            logger.info(f"Few categories ({num_categories}) - using pie chart")
            return self._create_pie_chart(kpi, data)
        
        # > 15 categories: Horizontal bar, top 20
        elif num_categories > 15:
            # Sort and take top 20
            data_sorted = data.nlargest(20, value_col)
            
            fig = go.Figure(go.Bar(
                x=data_sorted[value_col],
                y=data_sorted[category_col],
                orientation='h',
                text=data_sorted[value_col],
                texttemplate='%{text:,.0f}',
                textposition='auto',
                marker={
                    'color': data_sorted[value_col],
                    'colorscale': 'Blues',
                    'showscale': False
                },
                hovertemplate='%{y}<br>%{x:,.0f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f"{kpi.name} (Top 20)",
                xaxis_title=value_col,
                yaxis_title=category_col,
                height=self.TALL_CHART_HEIGHT,
                template=self.theme,
                yaxis={'categoryorder': 'total ascending'}
            )
        
        # 5-15 categories: Vertical bar
        else:
            fig = go.Figure(go.Bar(
                x=data[category_col],
                y=data[value_col],
                text=data[value_col],
                texttemplate='%{text:,.0f}',
                textposition='auto',
                marker={
                    'color': self.PRIMARY_COLOR,
                    'line': {'color': 'white', 'width': 1}
                },
                hovertemplate='%{x}<br>%{y:,.0f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=kpi.name,
                xaxis_title=category_col,
                yaxis_title=value_col,
                height=self.DEFAULT_HEIGHT,
                template=self.theme
            )
        
        return fig
    
    def _create_pie_chart(self, kpi: KPI, data: pd.DataFrame) -> go.Figure:
        """
        Create pie chart (ONLY for < 5 categories - Q4B)
        """
        # Identify category and value columns
        category_col = data.select_dtypes(include='object').columns[0]
        value_col = data.select_dtypes(include='number').columns[0]
        
        # Limit to 5 categories (should already be filtered, but double-check)
        if len(data) > 5:
            logger.warning(f"Too many categories for pie chart: {len(data)}")
            data = data.nlargest(5, value_col)
        
        fig = go.Figure(go.Pie(
            labels=data[category_col],
            values=data[value_col],
            textinfo='label+percent',
            textposition='auto',
            hovertemplate='%{label}<br>%{value:,.0f} (%{percent})<extra></extra>',
            marker={
                'colors': self.COLOR_SCHEME[:len(data)],
                'line': {'color': 'white', 'width': 2}
            }
        ))
        
        fig.update_layout(
            title=kpi.name,
            height=self.DEFAULT_HEIGHT,
            template=self.theme,
            showlegend=True
        )
        
        return fig
    
    def _create_table(self, kpi: KPI, data: pd.DataFrame) -> go.Figure:
        """
        Create data table for detailed view
        """
        # Format numeric columns
        formatted_data = data.copy()
        for col in formatted_data.select_dtypes(include='number').columns:
            formatted_data[col] = formatted_data[col].apply(
                lambda x: f"{x:,.0f}" if pd.notna(x) else ""
            )
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(formatted_data.columns),
                fill_color=self.PRIMARY_COLOR,
                font=dict(color='white', size=14),
                align='left'
            ),
            cells=dict(
                values=[formatted_data[col] for col in formatted_data.columns],
                fill_color='white',
                align='left',
                font=dict(size=12),
                height=30
            )
        )])
        
        fig.update_layout(
            title=kpi.name,
            height=min(500, 100 + len(data) * 35),
            template=self.theme,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        return fig
    
    def _create_empty_chart(self, title: str) -> go.Figure:
        """Create placeholder chart for empty/error state"""
        fig = go.Figure()
        
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color='gray')
        )
        
        fig.update_layout(
            title=title,
            height=self.DEFAULT_HEIGHT,
            template=self.theme,
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        
        return fig
    
    def _get_number_format(self, value: float) -> str:
        """
        Determine appropriate number format
        
        Args:
            value: Sample value
            
        Returns:
            Plotly format string
        """
        if pd.isna(value):
            return ',.0f'
        
        abs_value = abs(value)
        
        # Large numbers: use K, M, B suffixes
        if abs_value >= 1e9:
            return ',.2s'  # Billions
        elif abs_value >= 1e6:
            return ',.2s'  # Millions
        elif abs_value >= 1e3:
            return ',.0f'  # Thousands with commas
        elif abs_value >= 1:
            return ',.2f'  # Decimals
        else:
            return '.4f'   # Small decimals
    
    def auto_detect_chart_type(self, data: pd.DataFrame) -> str:
        """
        Automatically detect appropriate chart type from data
        
        Args:
            data: Query result data
            
        Returns:
            Chart type: 'metric', 'line', 'bar', 'pie', or 'table'
        """
        if data.empty:
            return 'metric'
        
        num_rows = len(data)
        num_cols = len(data.columns)
        
        # Single value: metric card
        if num_rows == 1 and num_cols == 1:
            return 'metric'
        
        # Check for date columns (time series)
        date_cols = data.select_dtypes(include=['datetime64', 'datetime']).columns
        has_date = len(date_cols) > 0
        
        # Check for numeric columns
        numeric_cols = data.select_dtypes(include='number').columns
        has_numeric = len(numeric_cols) > 0
        
        # Check for categorical columns
        categorical_cols = data.select_dtypes(include='object').columns
        has_categorical = len(categorical_cols) > 0
        
        # Time series data: line chart
        if has_date and has_numeric:
            return 'line'
        
        # Categorical with numeric: bar or pie
        if has_categorical and has_numeric and num_rows <= 100:
            if num_rows < 5:
                return 'pie'
            else:
                return 'bar'
        
        # Many rows: table
        if num_rows > 100:
            return 'table'
        
        # Default: table
        return 'table'


class ChartStyler:
    """
    Provides consistent styling across charts
    """
    
    @staticmethod
    def apply_theme(fig: go.Figure, theme: str = 'plotly_white') -> go.Figure:
        """Apply consistent theme to figure"""
        fig.update_layout(template=theme)
        return fig
    
    @staticmethod
    def add_branding(fig: go.Figure, text: str = "ChatDB") -> go.Figure:
        """Add subtle branding to chart"""
        fig.add_annotation(
            text=text,
            xref="paper", yref="paper",
            x=1, y=0,
            showarrow=False,
            font=dict(size=10, color='lightgray'),
            xanchor='right',
            yanchor='bottom'
        )
        return fig
    
    @staticmethod
    def optimize_for_dashboard(fig: go.Figure) -> go.Figure:
        """Optimize chart for dashboard display"""
        fig.update_layout(
            margin=dict(l=40, r=40, t=60, b=40),
            font=dict(size=12),
            hoverlabel=dict(
                bgcolor="white",
                font_size=12
            )
        )
        return fig


class VisualizationError(Exception):
    """Base exception for visualization errors"""
    pass


class InvalidDataError(VisualizationError):
    """Raised when data format is invalid for chart type"""
    pass
