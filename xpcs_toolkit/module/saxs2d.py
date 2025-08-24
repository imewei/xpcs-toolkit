def plot(
    xfile,
    pg_hdl=None,
    plot_type="log",
    cmap="jet",
    rotate=False,
    autolevel=False,
    autorange=False,
    vmin=None,
    vmax=None,
):
    # Return early if no plot handler provided
    if pg_hdl is None:
        return rotate
        
    center = (xfile.bcx, xfile.bcy)
    if plot_type == "log":
        img = xfile.saxs_2d_log
    else:
        img = xfile.saxs_2d

    if cmap is not None and hasattr(pg_hdl, 'set_colormap'):
        pg_hdl.set_colormap(cmap)

    prev_img = getattr(pg_hdl, 'image', None)
    shape_changed = prev_img is None or prev_img.shape != img.shape
    do_autorange = autorange or shape_changed

    # Initialize view_range to None
    view_range = None
    
    # Save view range if keeping it and view attribute exists
    if not do_autorange and hasattr(pg_hdl, 'view') and pg_hdl.view is not None:
        view_range = pg_hdl.view.viewRange()

    # Set new image if method exists
    if hasattr(pg_hdl, 'setImage'):
        pg_hdl.setImage(img, autoLevels=autolevel, autoRange=do_autorange)

    # Restore view range if we have it and skipped auto-ranging
    if not do_autorange and view_range is not None and hasattr(pg_hdl, 'view') and pg_hdl.view is not None:
        pg_hdl.view.setRange(xRange=view_range[0], yRange=view_range[1], padding=0)

    # Restore levels if needed and method exists
    if not autolevel and vmin is not None and vmax is not None and hasattr(pg_hdl, 'setLevels'):
        pg_hdl.setLevels(vmin, vmax)

    # Restore intensity levels (if needed) - removing duplicate code
    # The above condition already handles this case

    if center is not None and hasattr(pg_hdl, 'add_roi'):
        pg_hdl.add_roi(sl_type="Center", center=center, label="Center")

    return rotate
