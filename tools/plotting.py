# =============================================================================
# Plotting Functions   
# =============================================================================
#
# Description:
#   Functions for plotting spatial data with the cartopy library. 
#
# Author:
#   Luke Colosi
#
# Created:
#   2026-08-12
# =============================================================================

# Import libraries 
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# --- Coastline and Land Mask Function --- #
def set_coastlines(ax, projection, resolution, lon_min, lon_max, lat_min, lat_max):

    """
    set_subplots(ax, projection, resolution, lon_min, lon_max, lat_min, lat_max)

        Function for setting the spatial limits of a cartopy figure and plotting the 
        coastline and land mask. 

        Parameters
        ----------
        ax         : Geospatial axes for the subplot (cartopy object).
        projection : Cartopy map projection. 
        resolution : Specifies the resolution of the coastline map. 
                     Options include: '110m', '50m', '10m'
        lon_min    : Minimum extent for longitude on the scale from -180 to 179
        lon_max    : Maximum extent for longitude on the scale from -180 to 179
        lat_min    : Minimum extent for latitude on the scale from -90 to 89
        lat_max    : Maximum extent for latitude on the scale from -90 to 89

        Returns
        -------
        No objects returned. A geospatial map with desired longitude and latitude
        extent with coastlines and land.

        Libraries necessary to run function
        -----------------------------------
        import cartopy.feature as cfeature
    """

    # Set extents of map
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], projection)

    # Plot coastlines and land
    ax.coastlines(resolution=resolution)
    ax.add_feature(
        cfeature.NaturalEarthFeature("physical", "land", resolution, facecolor="Gray")
    )

    return


# --- Grid Lines and Tickmarks Function --- #
def set_grid_ticks(
    ax,
    xticks,
    yticks,
    xlabels=True,
    ylabels=True,
    grid=True,
    fontsize=12,
    color="k",
    lw=0.5,
    ls="--",
    alpha=0.25,
):
    """
    Set geographic tick marks, tick labels, and grid lines.

    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxes
        Geospatial axes to format.
    xticks : array-like
        Longitude tick locations.
    yticks : array-like
        Latitude tick locations.
    xlabels : bool, optional
        If True, display longitude tick labels.
    ylabels : bool, optional
        If True, display latitude tick labels.
    grid : bool, optional
        If True, display grid lines.
    fontsize : int or float, optional
        Font size of the longitude and latitude tick labels.
    color : str, optional
        Color of the grid lines.
    lw : int or float, optional
        Width of the grid lines.
    ls : str, optional
        Line style of the grid lines.
    alpha : float, optional
        Transparency of the grid lines.

    Returns
    -------
    None
    """

    # Set longitude and latitude ticks
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())

    # Format longitude and latitude tick labels
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    # Set tick-label properties
    ax.tick_params(
        axis="both",
        labelsize=fontsize,
        colors="k",
        labelbottom=xlabels,
        labelleft=ylabels,
    )

    # Set grid lines
    if grid:
        ax.grid(
            lw=lw,
            ls=ls,
            color=color,
            alpha=alpha,
        )

    return


# --- Colorbar Function --- #
def set_cbar(
    cs,
    cax,
    fig,
    orientation="vertical",
    extend="neither",
    label="",
    fontsize=12,
    ticks=None,
    tick_direction="out",
    tick_length=4,
    invert=False,
):
    """
    Create and customize a colorbar.

    Parameters
    ----------
    cs : matplotlib plotting object
        Plot object containing the graphical representation of the data.
    cax : matplotlib.axes.Axes
        Axes in which to draw the colorbar.
    fig : matplotlib.figure.Figure
        Figure to which the colorbar is attached.
    orientation : {"vertical", "horizontal"}, optional
        Orientation of the colorbar.
    extend : {"neither", "both", "min", "max"}, optional
        Direction in which the colorbar extends beyond its limits.
    label : str, optional
        Colorbar label.
    fontsize : int or float, optional
        Font size of the colorbar label and tick labels.
    ticks : array-like, optional
        Locations of the colorbar ticks.
    tick_direction : {"in", "out", "inout"}, optional
        Direction of the colorbar tick marks.
    tick_length : int or float, optional
        Length of the colorbar tick marks in points.
    invert : bool, optional
        If True, invert the colorbar axis.

    Returns
    -------
    cbar : matplotlib.colorbar.Colorbar
        Created colorbar object.
    """

    # Create colorbar
    cbar = fig.colorbar(
        cs,
        cax=cax,
        orientation=orientation,
        extend=extend,
        ticks=ticks,
        label=label,
    )

    # Set font size for labels 
    cbar.ax.xaxis.label.set_size(fontsize)
    cbar.ax.yaxis.label.set_size(fontsize)

    # Set tick properties
    cbar.ax.tick_params(
        labelsize=fontsize,
        direction=tick_direction,
        length=tick_length,
    )

    # Invert colorbar axis
    if invert:
        if orientation == "vertical":
            cbar.ax.invert_yaxis()
        else:
            cbar.ax.invert_xaxis()

    return cbar

# --- Figure corner labeling --- #
def add_corner_label(ax, pos, label, fontsize=12):

    """
    Add a labeled text box to a specified corner of an axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object on which to place the label.
    pos : tuple of float
        The (x, y) location in axes coordinates where the label will be placed.
        Both values should be between 0 and 1, where (0, 0) is the lower-left
        corner and (1, 1) is the upper-right corner of the axes.
    label : str
        The text content of the label.
    fontsize : int, optional
        Font size of the label text. Default is 12.

    Returns
    -------
    None
        The function adds the label directly to the axes.
    """

    # Place text in lower left corner inside the axes
    ax.text(
        pos[0], pos[1], label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight='normal',
        va='center', ha='center',
        bbox=dict(
            boxstyle='square,pad=0.3',
            facecolor=(1, 1, 1, 0.6), 
            edgecolor='black',
            linewidth=1
        )
    )

    return