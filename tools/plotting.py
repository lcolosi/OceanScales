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
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib.transforms as transforms

# --- Coastline and Land Mask Function --- #
def set_coastlines(
    ax, 
    projection, 
    resolution, 
    lon_min, 
    lon_max, 
    lat_min, 
    lat_max,
):

    """
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
def add_corner_label(
    ax, 
    pos, 
    label, 
    fontsize=12,
):
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

# --- Figure corner labeling --- #
def month_fmt(
    x,
    pos,
):
    """
    Custom formatter function for labeling months on a Matplotlib time axis.

    Parameters
    ----------
    x : float
        The x-axis value representing time in Matplotlib's internal date format 
        (i.e., days since 0001-01-01 UTC, plus fractions of a day).

    pos : int
        The tick position index (required by Matplotlib's formatter interface, 
        but not used in this function).

    Returns
    -------
    label : str
        A formatted string for the x-axis tick label — either the first letter of 
        the month or, for January, the letter 'J' followed by the year on a new line.
    """

    # Convert the numeric x-axis value into a datetime object
    dt = mdates.num2date(x)

    # If the month is January, return 'J' with the year printed below (newline)
    if dt.month == 1:
        return f"J\n{dt.year}"  
    
    # For all other months, return only the first letter of the abbreviated month name
    else:
        return dt.strftime('%b')[0]


# --- Length Scale Bar --- #
def add_scalebar(
    ax, 
    length_km=50, 
    location=(0.15, 0.07),
    linewidth=2, 
    text_kwargs=None
):
    """
    Add a horizontal scale bar to a Cartopy GeoAxes.

    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxes
        The axes to draw the scale bar on.
    length_km : float
        Length of the scale bar in kilometers.
    location : tuple of float
        Location of the center of the scale bar in *axes* coordinates
        (x, y), both between 0 and 1.
    linewidth : float
        Line width of the scale bar.
    text_kwargs : dict, optional
        Extra kwargs passed to ax.text (fontsize, weight, etc.).
    """

    # Set key word arguments when none are given 
    if text_kwargs is None:
        text_kwargs = dict(fontsize=10)

    # Get current map extent in data (PlateCarree) coordinates
    lon_min, lon_max, lat_min, lat_max = ax.get_extent(crs=ccrs.PlateCarree())

    # Choose the latitude where the bar will be drawn
    lat = lat_min + location[1] * (lat_max - lat_min)

    # Convert desired length (km) to degrees of longitude at that latitude
    km_per_deg_lon = 111.32 * np.cos(np.deg2rad(lat))
    dlon = (length_km / km_per_deg_lon)

    # Choose center longitude based on axes location
    lon_center = lon_min + location[0] * (lon_max - lon_min)
    lon_left   = lon_center - dlon / 2
    lon_right  = lon_center + dlon / 2

    # Small vertical tick height in degrees
    dtick = 0.01 * (lat_max - lat_min)

    # Horizontal bar
    ax.plot([lon_left, lon_right], [lat, lat],
            transform=ccrs.PlateCarree(),
            color='k', linewidth=linewidth, solid_capstyle='butt')

    # Vertical ticks at ends (to make |-----|)
    ax.plot([lon_left, lon_left],
            [lat - dtick, lat + dtick],
            transform=ccrs.PlateCarree(),
            color='k', linewidth=linewidth)
    ax.plot([lon_right, lon_right],
            [lat - dtick, lat + dtick],
            transform=ccrs.PlateCarree(),
            color='k', linewidth=linewidth)

    # Label above the bar
    ax.text(lon_center, lat + 1.8 * dtick,
            f"{int(length_km)} km",
            ha='center', va='bottom',
            transform=ccrs.PlateCarree(),
            **text_kwargs)


# --- Plot Top Axis Markers --- #
def add_x_axis_marker(
    ax,
    x_pos,
    marker,
    label,
    y_marker=1.01,
    y_text=1.07,
    ms=6,
    x_text_offset_pts=0.0,
    markerfacecolor="white",
    markeredgecolor="k",
    fontsize=10,
):
    """
    Add a marker above the top x-axis with a text label.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which to draw the marker and label.
    x_pos : float
        Marker position along the x-axis in data coordinates.
    marker : str
        Matplotlib marker symbol.
    label : str
        Text label to place above the marker.
    y_marker : float, optional
        Vertical marker position in axes coordinates. Values greater
        than 1 place the marker above the plotting area. Default is 1.01.
    y_text : float, optional
        Vertical text position in axes coordinates. Default is 1.07.
    ms : float, optional
        Marker size. Default is 6.
    x_text_offset_pts : float, optional
        Horizontal text offset from the marker in points. Positive values
        shift the text right and negative values shift it left.
        Default is 0.0.
    markerfacecolor : str or tuple, optional
        Marker face color. Default is "white".
    markeredgecolor : str or tuple, optional
        Marker edge color. Default is "k".
    fontsize : float, optional
        Text label font size. Default is 10.

    Returns
    -------
    None
    """

    # Use data coordinates for x and axes coordinates for y
    marker_transform = transforms.blended_transform_factory(
        ax.transData,
        ax.transAxes,
    )

    # Plot marker above the top axis
    ax.plot(
        x_pos,
        y_marker,
        marker=marker,
        linestyle="None",
        markersize=ms,
        markerfacecolor=markerfacecolor,
        markeredgecolor=markeredgecolor,
        transform=marker_transform,
        clip_on=False,
        zorder=10,
    )

    # Apply an optional horizontal offset to the text label
    text_transform = transforms.offset_copy(
        marker_transform,
        fig=ax.figure,
        x=x_text_offset_pts,
        y=0.0,
        units="points",
    )

    # Add text label above the marker
    ax.text(
        x_pos,
        y_text,
        label,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        transform=text_transform,
        clip_on=False,
        zorder=10,
    )


# --- Status Message --- # 
def status(message):
    """
    Print a timestamped status message to the console.

    Parameters
    ----------
    message : str
        Status message to display.

    Returns
    -------
    None
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)