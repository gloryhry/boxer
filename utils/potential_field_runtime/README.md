# Potential-field source attribution

The `roomseg/` files are the production-mode runtime from Room Atlas
(`room_instance_seg`) at commit `7fd3926aa801a48be3622ae185c16e12c6a7e0da`. The source repository
records its upstream as [uturkbey's Python reproduction](https://github.com/uturkbey/Room-Segmentation-in-Point-Clouds)
at commit `cb45f376634f30cf3f9fee640af47756145357aa`.

The method is described by D. Bobkov, M. Kiechle, S. Hilsenbeck, and E. Steinbach,
*Room Segmentation in 3D Point Clouds Using Anisotropic Potential Fields*, IEEE
ICME 2017. [Paper and project page](https://dbobkov.github.io/room-segmentation/) ·
[DOI 10.1109/ICME.2017.8019484](https://doi.org/10.1109/ICME.2017.8019484)

The Boxer adapter selects Room Atlas's supported single-height sampling modes.
The copy includes the seven active Python modules. It omits the historical
experimental modules, demo data, render assets, and standalone viewer.

The upstream Python repository does not publish an explicit license. Confirm
the applicable redistribution permission before distributing these files.
