"""
Reusable plot styling configuration for publication-ready visualizations.

Usage:
    from utils.plot_styling import PlotStyle, apply_style
    
    # Apply global matplotlib styling
    apply_style()
    
    # Access colors and utilities
    colors = PlotStyle.COLORS
    gradient = PlotStyle.get_gradient_colors(n=10)
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


class PlotStyle:
    """
    Centralized styling configuration for matplotlib plots.
    
    Color palette: Nordic Ocean gradient (Baltic Blue → Tea Green).
    Designed for white backgrounds with clean, modern aesthetics.
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # COLOR PALETTE - Nordic Ocean Theme
    # ═══════════════════════════════════════════════════════════════════════════
    
    COLORS = {
        # Primary gradient (Baltic Blue → Tea Green)
        'baltic_blue': '#22577A',       # Deep oceanic blue - darkest
        'tropical_teal': '#38A3A5',     # Vivid teal
        'mint_leaf': '#57CC99',         # Refreshing mint
        'light_green': '#80ED99',       # Soft spring green
        'tea_green': '#C7F9CC',         # Pale creamy green - lightest
        
        # Mapped aliases for code compatibility
        'primary': '#38A3A5',           # Tropical Teal
        'primary_light': '#57CC99',     # Mint Leaf
        'primary_dark': '#22577A',      # Baltic Blue
        'primary_pale': '#C7F9CC',      # Tea Green
        
        # Accent colors (contrasting warm tones)
        'accent': '#22577A',            # Baltic Blue (for reference lines)
        'accent_light': '#38A3A5',      # Tropical Teal
        'accent_dark': '#1A4159',       # Darker blue for emphasis
        
        # Secondary (using the lighter greens)
        'secondary': '#80ED99',         # Light Green
        'secondary_light': '#C7F9CC',   # Tea Green
        
        # Neutrals (slate tones)
        'text_dark': '#1E293B',         # Slate-800
        'text_medium': '#64748B',       # Slate-500
        'text_light': '#94A3B8',        # Slate-400
        
        # Backgrounds & borders
        'bg_white': '#FFFFFF',
        'bg_light': '#F8FAFC',          # Slate-50
        'border': '#E2E8F0',            # Slate-200
        'grid': '#F1F5F9',              # Slate-100
        
        # Semantic colors
        'success': '#57CC99',           # Mint Leaf
        'warning': '#F59E0B',           # Amber-500
        'error': '#EF4444',             # Red-500
    }
    
    # Gradient sequence for multi-bar charts (dark to light)
    GRADIENT_SEQUENCE = ['#22577A', '#38A3A5', '#57CC99', '#80ED99', '#C7F9CC']
    
    # ═══════════════════════════════════════════════════════════════════════════
    # AUTHOR COLOR PALETTE - Nordic Ocean Extended Theme
    # ═══════════════════════════════════════════════════════════════════════════
    # 
    # A cohesive palette for distinguishing authors/categories that complements
    # the Nordic Ocean gradient. Colors are designed for:
    # - Clear visual distinction between authors
    # - Accessibility and readability
    # - Aesthetic harmony with the main theme
    #
    AUTHOR_COLORS = [
        # Primary Nordic Ocean colors (base palette)
        '#22577A',   # Baltic Blue - deep oceanic blue
        '#B8956B',   # Driftwood Tan - warm sandy brown (distinct from blues/greens)
        '#57CC99',   # Mint Leaf - refreshing mint
        
        # Coastal Sunset tones (warm complementary)
        '#E07A5F',   # Terra Cotta - warm coral red
        '#8E7C93',   # Stone Mauve - cool gray-purple (distinct from reds/yellows)
        '#F4D35E',   # Sunlit Gold - soft warm yellow
        
        # Nordic Forest tones (earthy greens)
        '#2D5016',   # Pine Deep - dark forest green
        '#6B8E23',   # Olive Drab - moss green
        '#98C379',   # Sage Green - light forest
        
        # Arctic Sky tones (cool purples & grays)
        '#6C5B7B',   # Nordic Plum - muted purple
        '#9B8AA5',   # Lavender Mist - soft lavender
        '#3D5A80',   # Steel Blue - slate blue
        
        # Ocean Depths (darker blues & teals)
        '#1A4159',   # Deep Ocean - darker blue
        '#2A6B7C',   # Fjord Teal - deep teal
        '#4A7C8A',   # Storm Sea - medium blue-gray
        
        # Aurora tones (vibrant accents)
        '#C44569',   # Aurora Pink - vivid magenta
        '#FF6B6B',   # Coral Red - bright coral
        '#845EC2',   # Aurora Violet - rich purple
        
        # Additional distinguishable colors
        '#00896F',   # Emerald - bright teal-green
        '#2C698D',   # Ocean Blue - medium blue
    ]
    
    # Named author colors for specific assignments (optional)
    NAMED_AUTHOR_COLORS = {
        'author_1': '#22577A',   # Baltic Blue
        'author_2': '#E07A5F',   # Terra Cotta
        'author_3': '#57CC99',   # Mint Leaf
        'author_4': '#6C5B7B',   # Nordic Plum
        'author_5': '#F2A359',   # Sandy Orange
        'author_6': '#3D5A80',   # Steel Blue
        'author_7': '#98C379',   # Sage Green
        'author_8': '#C44569',   # Aurora Pink
    }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TYPOGRAPHY - Scientific Paper Standards
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def get_font_family() -> str:
        """
        Get the best available scientific font.
        
        Priority (for figures):
        1. Helvetica - Gold standard for scientific figures
        2. Liberation Sans - Free Helvetica clone (common on Linux)
        3. Arial - Windows equivalent
        4. Nimbus Sans - Another Helvetica clone
        5. DejaVu Sans - Fallback
        """
        try:
            available = [f.name for f in fm.fontManager.ttflist]
            # Scientific publication fonts in order of preference
            preferred = [
                'Helvetica',           # Industry standard
                'Helvetica Neue',      # Modern variant
                'Liberation Sans',     # Free Helvetica clone (Linux)
                'Nimbus Sans',         # PostScript Helvetica clone
                'Arial',               # Windows standard
                'TeX Gyre Heros',      # LaTeX Helvetica clone
                'DejaVu Sans',         # Good fallback
            ]
            return next((f for f in preferred if f in available), 'DejaVu Sans')
        except:
            return 'DejaVu Sans'
    
    @staticmethod
    def get_serif_font() -> str:
        """
        Get the best available serif font for body text style.
        
        Priority:
        1. Times New Roman - Most common in journals
        2. Liberation Serif - Free Times clone
        3. STIX Two Text - Scientific publishing font
        4. Computer Modern - LaTeX default
        """
        try:
            available = [f.name for f in fm.fontManager.ttflist]
            preferred = [
                'Times New Roman',
                'Liberation Serif',
                'STIX Two Text',
                'CMU Serif',           # Computer Modern
                'TeX Gyre Termes',     # LaTeX Times clone
                'Nimbus Roman',
                'DejaVu Serif',
            ]
            return next((f for f in preferred if f in available), 'DejaVu Serif')
        except:
            return 'DejaVu Serif'
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GRADIENT UTILITIES
    # ═══════════════════════════════════════════════════════════════════════════
    
    @classmethod
    def create_gradient_cmap(cls, color1: str = None, color2: str = None) -> mcolors.LinearSegmentedColormap:
        """Create a linear gradient colormap between two colors."""
        c1 = color1 or cls.COLORS['baltic_blue']
        c2 = color2 or cls.COLORS['tea_green']
        return mcolors.LinearSegmentedColormap.from_list(
            'nordic_ocean', 
            [mcolors.to_rgb(c1), mcolors.to_rgb(c2)]
        )
    
    @classmethod
    def create_full_gradient_cmap(cls) -> mcolors.LinearSegmentedColormap:
        """Create colormap using all 5 palette colors for smoother gradients."""
        colors = [mcolors.to_rgb(c) for c in cls.GRADIENT_SEQUENCE]
        return mcolors.LinearSegmentedColormap.from_list('nordic_ocean_full', colors)
    
    @classmethod
    def get_gradient_colors(cls, n: int, reverse: bool = False) -> List[str]:
        """Get n colors from the Nordic Ocean gradient."""
        cmap = cls.create_full_gradient_cmap()
        colors = [mcolors.to_hex(cmap(i / max(n - 1, 1))) for i in range(n)]
        return colors[::-1] if reverse else colors
    
    @classmethod
    def get_categorical_colors(cls, n: int) -> List[str]:
        """Get n distinct categorical colors for different categories."""
        # Use the 5 main palette colors first
        base_colors = cls.GRADIENT_SEQUENCE.copy()
        if n <= len(base_colors):
            return base_colors[:n]
        # If need more, interpolate from the gradient
        return cls.get_gradient_colors(n)
    
    @classmethod
    def get_author_colors(cls, n: int) -> List[str]:
        """
        Get n distinct colors for author/category visualization.
        
        Uses the AUTHOR_COLORS palette which is designed for maximum
        distinguishability while maintaining Nordic Ocean aesthetic.
        
        Args:
            n: Number of colors needed
            
        Returns:
            List of hex color strings
        """
        if n <= len(cls.AUTHOR_COLORS):
            return cls.AUTHOR_COLORS[:n]
        
        # If more colors needed, cycle through and slightly modify
        colors = cls.AUTHOR_COLORS.copy()
        while len(colors) < n:
            # Add slight variations of existing colors
            base_idx = len(colors) % len(cls.AUTHOR_COLORS)
            base_color = cls.AUTHOR_COLORS[base_idx]
            # Lighten or darken the color
            modified = cls._modify_color(base_color, factor=0.8 + 0.4 * (len(colors) // len(cls.AUTHOR_COLORS)))
            colors.append(modified)
        return colors[:n]
    
    @classmethod
    def get_author_color_by_index(cls, index: int) -> str:
        """
        Get a specific author color by index.
        
        Args:
            index: Index into the author color palette
            
        Returns:
            Hex color string
        """
        return cls.AUTHOR_COLORS[index % len(cls.AUTHOR_COLORS)]
    
    @staticmethod
    def _modify_color(hex_color: str, factor: float = 0.8) -> str:
        """
        Modify a hex color by a brightness factor.
        
        Args:
            hex_color: Hex color string (e.g., '#22577A')
            factor: Brightness factor (< 1 darkens, > 1 lightens)
            
        Returns:
            Modified hex color string
        """
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        
        r = min(255, max(0, int(r * factor)))
        g = min(255, max(0, int(g * factor)))
        b = min(255, max(0, int(b * factor)))
        
        return f'#{r:02x}{g:02x}{b:02x}'
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FORMATTING UTILITIES
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def format_number(num: float, precision: int = 1) -> str:
        """Format numbers with K/M suffix for thousands/millions."""
        if abs(num) >= 1_000_000:
            return f'{num/1_000_000:.{precision}f}M'
        if abs(num) >= 1_000:
            return f'{num/1_000:.{precision}f}K'
        if isinstance(num, float) and not num.is_integer():
            return f'{num:.{precision}f}'
        return f'{int(num)}'
    
    @staticmethod
    def truncate_label(label: str, max_len: int = 15) -> str:
        """Truncate long labels with ellipsis."""
        label = str(label).replace('_', ' ').title()
        return label[:max_len-2] + '…' if len(label) > max_len else label
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PLOT ELEMENT STYLING
    # ═══════════════════════════════════════════════════════════════════════════
    
    @classmethod
    def style_axis(cls, ax, title: str = None, xlabel: str = None, ylabel: str = None,
                   grid_axis: str = 'y', title_loc: str = 'left'):
        """Apply consistent styling to an axis."""
        if title:
            ax.set_title(title, fontsize=13,  
                        color=cls.COLORS['text_dark'], pad=15, loc=title_loc)
        if xlabel:
            ax.set_xlabel(xlabel, color=cls.COLORS['text_medium'], fontsize=10)
        if ylabel:
            ax.set_ylabel(ylabel, color=cls.COLORS['text_medium'], fontsize=10)
        
        # Grid
        if grid_axis:
            if 'x' in grid_axis:
                ax.xaxis.grid(True, linestyle='-', alpha=0.7, color=cls.COLORS['grid'])
            if 'y' in grid_axis:
                ax.yaxis.grid(True, linestyle='-', alpha=0.7, color=cls.COLORS['grid'])
        ax.set_axisbelow(True)
        
        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    @classmethod
    def add_bar_labels(cls, ax, bars, values, fmt_func=None, offset_ratio: float = 0.02,
                       position: str = 'top', fontsize: int = 8):
        """Add value labels to bar charts."""
        fmt_func = fmt_func or cls.format_number
        max_val = max(values) if len(values) > 0 else 1
        
        for bar, val in zip(bars, values):
            if position == 'top':
                x = bar.get_x() + bar.get_width() / 2
                y = bar.get_height()
                va = 'bottom'
                ha = 'center'
            elif position == 'end':  # For horizontal bars
                x = bar.get_width() + max_val * offset_ratio
                y = bar.get_y() + bar.get_height() / 2
                va = 'center'
                ha = 'left'
            else:
                continue
                
            ax.text(x, y, fmt_func(val), ha=ha, va=va,
                   fontsize=fontsize, fontweight='medium',
                   color=cls.COLORS['text_medium'])
    
    @classmethod
    def add_metadata_badge(cls, ax, text: str, loc: str = 'lower right'):
        """Add a styled metadata badge to the plot."""
        positions = {
            'lower right': (0.98, 0.02, 'right', 'bottom'),
            'lower left': (0.02, 0.02, 'left', 'bottom'),
            'upper right': (0.98, 0.98, 'right', 'top'),
            'upper left': (0.02, 0.98, 'left', 'top'),
        }
        x, y, ha, va = positions.get(loc, positions['lower right'])
        
        ax.text(x, y, text, transform=ax.transAxes,
               ha=ha, va=va, fontsize=8, color=cls.COLORS['text_light'],
               style='italic',
               bbox=dict(boxstyle='round,pad=0.4', 
                        facecolor=cls.COLORS['bg_white'],
                        edgecolor=cls.COLORS['border'], 
                        alpha=0.9))


def apply_style():
    """Apply global matplotlib styling. Call this at the start of your script."""
    
    style = PlotStyle()
    
    plt.rcParams.update({
        # Font
        'font.family': style.get_font_family(),
        'font.size': 10,
        
        # Axes titles & labels
        'axes.titlesize': 13,
        #'axes.titleweight': 'bold',
        'axes.labelsize': 10,
        'axes.labelweight': 'medium',
        'axes.labelcolor': style.COLORS['text_dark'],
        
        # Axes appearance
        'axes.edgecolor': style.COLORS['border'],
        'axes.linewidth': 0,
        'axes.facecolor': style.COLORS['bg_white'],
        'axes.spines.right': False,
        'axes.spines.top': False,
        'axes.spines.left': False,
        'axes.spines.bottom': False,
        
        # Ticks
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.color': style.COLORS['text_medium'],
        'ytick.color': style.COLORS['text_medium'],
        'xtick.major.size': 0,
        'ytick.major.size': 0,
        
        # Grid
        'grid.color': style.COLORS['grid'],
        'grid.linewidth': 0.8,
        'grid.alpha': 0.8,
        
        # Figure
        'figure.facecolor': style.COLORS['bg_white'],
        'savefig.facecolor': style.COLORS['bg_white'],
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        
        # Legend
        'legend.frameon': False,
        'legend.fontsize': 9,
    })


def create_figure(nrows: int = 1, ncols: int = 1, figsize: Tuple[float, float] = None,
                  **kwargs) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a figure with consistent styling.
    
    Args:
        nrows: Number of subplot rows
        ncols: Number of subplot columns
        figsize: Figure size (width, height) in inches
        **kwargs: Additional arguments passed to plt.subplots
        
    Returns:
        fig, axes tuple
    """
    if figsize is None:
        # Smart default sizing
        base_w, base_h = 5.5, 4.5
        figsize = (base_w * ncols + 0.5 * (ncols - 1), 
                   base_h * nrows + 0.5 * (nrows - 1))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    fig.patch.set_facecolor(PlotStyle.COLORS['bg_white'])
    
    return fig, axes

