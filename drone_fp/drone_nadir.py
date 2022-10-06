from __future__ import division
from osgeo import gdal, ogr, osr
import os, sys
import json
import geojson
import fnmatch
from shapely import affinity
import shapely.geometry
from shapely.geometry import shape, LineString, Point, Polygon
from shapely.ops import cascaded_union, transform
import argparse
import exiftool
import datetime
from operator import itemgetter
import geopandas as gp
import fiona.crs
from functools import partial
import pyproj
import math
from math import *
import ntpath
from geojson_rewind import rewind
from progress.bar import Bar

from vincenty import vincenty_inverse

parser = argparse.ArgumentParser(description="Input Mission JSON File")
parser.add_argument("-i", "--indir", help="Input directory", required=True)
parser.add_argument("-o", "--dest", help="Output directory", required=True)
parser.add_argument("-g", "--nonst", help="geoTIFF output", required=True)
args = parser.parse_args()
indir = args.indir
outdir = args.dest
geo_tiff = args.nonst
now = datetime.datetime.now()
file_name = "M_" + now.strftime("%Y-%m-%d_%H-%M") + ".json"


class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class errors_msn:
    list = []


def main():
    files = find_file(indir)
    read_exif(files)


def create_georaster(tags):
    # print(tags)
    """
    :param tags:
    :return:
    """
    out_out = ntpath.dirname(indir + "/outputs/")
    print("out dir", out_out)
    if not os.path.exists(out_out):
        os.makedirs(out_out)
    bar = Bar('Creating GeoTIFFs', max=len(tags))

    for tag in iter(tags):

        coords = tag['geometry']['coordinates'][0]
        # lonlat = coords[0]
        pt0 = coords[0][0], coords[0][1]
        pt1 = coords[1][0], coords[1][1]
        pt2 = coords[2][0], coords[2][1]
        pt3 = coords[3][0], coords[3][1]

        # print("OMGOMG", poly)
        props = tag['properties']
        # print("PROPS", props)
        # print(props)
        file_in = indir + "/images/" + props['File_Name']
        # print("file In", file_in)
        new_name = ntpath.basename(file_in[:-3]) + 'tif'
        dst_filename = out_out + "/" + new_name
        ds = gdal.Open(file_in, 0)
        gt = ds.GetGeoTransform()
        cols = ds.RasterXSize
        rows = ds.RasterYSize
        ext = GetExtent(gt, cols, rows)
        ext0 = ext[0][0], ext[0][1]
        ext1 = ext[1][0], ext[1][1]
        ext2 = ext[2][0], ext[2][1]
        ext3 = ext[3][0], ext[3][1]

        gcp_string = '-gcp {} {} {} {} ' \
                     '-gcp {} {} {} {} ' \
                     '-gcp {} {} {} {} ' \
                     '-gcp {} {} {} {}'.format(ext0[0], ext0[1],
                                               pt2[0], pt2[1],
                                               ext1[0], ext1[1],
                                               pt3[0], pt3[1],
                                               ext2[0], ext2[1],
                                               pt0[0], pt0[1],
                                               ext3[0], ext3[1],
                                               pt1[0], pt1[1])

        gcp_items = filter(None, gcp_string.split("-gcp"))
        gcp_list = []
        for item in gcp_items:
            pixel, line, x, y = map(float, item.split())
            z = 0
            gcp = gdal.GCP(x, y, z, pixel, line)
            gcp_list.append(gcp)

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        wkt = srs.ExportToWkt()
        ds = gdal.Translate(dst_filename, ds, outputSRS=wkt, GCPs=gcp_list, noData=0)
        ds = None
        bar.next()
    bar.finish()
    return


def GetExtent(gt,cols,rows):
    ''' Return list of corner coordinates from a geotransform
        @type gt:   C{tuple/list}
        @param gt: geotransform
        @type cols:   C{int}
        @param cols: number of columns in the dataset
        @type rows:   C{int}
        @param rows: number of rows in the dataset
        @rtype:    C{[float,...,float]}
        @return:   coordinates of each corner
    '''
    ext=[]
    xarr=[0,cols]
    yarr=[0,rows]

    for px in xarr:
        for py in yarr:
            x=gt[0]+(px*gt[1])+(py*gt[2])
            y=gt[3]+(px*gt[4])+(py*gt[5])
            ext.append([x,y])
            # print(x,y)
        yarr.reverse()
    return ext


def read_exif(files):
    print("How many files?", len(files))
    """
    :param files:
    :return:
    """
    exif_array = []
    filename = file_name
    bar = Bar('Reading EXIF Data', max=len(files))
    with exiftool.ExifToolHelper() as et:
        # print(color.BLUE + "Scanning image exif tags " + color.END)
        metadata = iter(et.get_metadata(files))
    for d in metadata:
        exif_array.append(d)
        bar.next()
    bar.finish()
    # print(color.BLUE + "Scanning images complete " + color.END)
    formatted = format_data(exif_array)
    writeOutputtoText(filename, formatted)
    print(color.GREEN + "Process Complete." + color.END)


def format_data(exif_array):
    """
    :param exif_array:
    :return:
    """
    sls = json.dumps(exif_array)
    #print(sls)
    print("How many records?", len(sls))
    try:
        exif_array.sort(key=itemgetter('EXIF:DateTimeOriginal'))
    except KeyError as e:
        pass
    feature_coll = dict(type="FeatureCollection", features=[])
    linecoords = []
    img_stuff = []
    datetime = ''
    sensor = ''
    sensor_make = ''
    i = 0
    bar = Bar('Creating GeoJSON', max=len(exif_array))
    for tags in iter(exif_array):
        i = i + 1
        for tag, val in tags.items():
            if tag in ('JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote'):
                exif_array.remove(tag)
        # print(tags)
        try:
            lat = float(tags['XMP:Latitude'])
            long = float(tags['XMP:Longitude'])
            imgwidth = tags['EXIF:ImageWidth']
            imghite = tags['EXIF:ImageHeight']
            alt = float(tags['XMP:RelativeAltitude'])
            dtgi = tags['EXIF:DateTimeOriginal']
        except KeyError as e:
            lat = float(tags['Composite:GPSLatitude'])
            long = float(tags['Composite:GPSLongitude'])
            imgwidth = tags['EXIF:ExifImageWidth']
            imghite = tags['EXIF:ExifImageHeight']
            alt = float(tags['XMP:RelativeAltitude'])
            dtgi = tags['EXIF:CreateDate']
        #coords = [long, lat, alt]
        coords = [lat, long, alt]
        linecoords.append(coords)
        try:
            ptProps = {"File_Name": tags['File:FileName'],
                       "Focal_Length": tags['EXIF:FocalLength'], "Date_Time": dtgi,
                       "Image_Width": imgwidth, "Image_Height": imghite,
                       "Heading": tags['XMP:FlightYawDegree'], "AbsoluteAltitude": alt,
                       "Relative_Altitude": tags['XMP:RelativeAltitude'],
                       "FlightRollDegree": tags['XMP:FlightRollDegree'], "FlightYawDegree": tags['XMP:FlightYawDegree'],
                       "FlightPitchDegree": tags['XMP:FlightPitchDegree'],
                       "GimbalRollDegree": tags['XMP:GimbalRollDegree'], "GimbalYawDegree": tags['XMP:GimbalYawDegree'],
                       "GimbalPitchDegree": tags['XMP:GimbalPitchDegree'],
                       "EXIF:DateTimeOriginal": tags['EXIF:DateTimeOriginal']}
        except KeyError as ke:
            ptProps = {"File_Name": tags['File:FileName'], "Exposure Time": tags['EXIF:ExposureTime'],
                       "Focal_Length": tags['EXIF:FocalLength'], "Date_Time": dtgi,
                       "Image_Width": imgwidth, "Image_Height": imghite, "Heading": tags['EXIF:GPSImgDirection'],
                       "AbsoluteAltitude": alt,
                       "Relative_Altitude": alt,
                       "EXIF:DateTimeOriginal": tags['EXIF:DateTimeOriginal']}
        if i == 1:
            datetime = dtgi
            sensor = tags['EXIF:Model']
            sensor_make = tags['EXIF:Make']
        img_over = dict(coords=coords, props=ptProps)
        img_stuff.append(img_over)
        ptGeom = dict(type="Point", coordinates=coords)
        points = dict(type="Feature", geometry=ptGeom, properties=ptProps)
        feature_coll['features'].append(points)
        bar.next()
        # gcp_info = long, lat, alt
    img_box = image_poly(img_stuff)
    tiles = img_box[0]
    # write_gcpList(gcp_info)
    if geo_tiff == 'y':
        create_georaster(tiles)
    else:
        print("no georasters.")
    aoi = img_box[1]
    lineGeom = dict(type="LineString", coordinates=linecoords)
    lines = dict(type="Feature", geometry=lineGeom, properties={})
    feature_coll['features'].insert(0, lines)
    mission_props = dict(date=datetime, platform="DJI Mavic 2 Pro", sensor_make=sensor_make, sensor=sensor)
    polys = dict(type="Feature", geometry=aoi, properties=mission_props)
    feature_coll['features'].insert(0, polys)
    for imps in tiles:
        feature_coll['features'].append(imps)
    bar.finish()
    return feature_coll


def writeOutputtoText(filename, file_list):
    """
    :param filename:
    :param file_list:
    :return:
    """
    dst_n = outdir + '/' + filename
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        os.chmod(outdir, 0o777)
    with open(dst_n, 'w') as outfile:
        geojson.dump(file_list, outfile, indent=4, sort_keys=False)
    print(color.GREEN + "GeoJSON Produced." + color.END)
    return


# def write_gcpList(gcp_arr):
#     """
#
#     :param filename:
#     :param file_list:
#     :return:
#     """
#
#     fileout = 'gcp_list.txt'
#     dst_n = outdir + '/' + fileout
#     with open(dst_n, 'w') as outfile:
#         geojson.dump(fileout, outfile)
#     print(color.GREEN + "GCP_list Produced." + color.END)
#     return


def image_poly(imgar):
    """
    :param imgar:
    :return:
    """
    polys = []
    over_poly = []
    darepo = ''
    bar = Bar('Plotting Image Bounds', max=len(imgar))
    # print("BAR", bar)
    for cent in iter(imgar):
        lat = float(cent['coords'][1])
        lng = float(cent['coords'][0])
        prps = cent['props']
        print("\nGeoprojecting image -> %s" % prps['File_Name'])
        wid= prps['Image_Width']
        hite = prps['Image_Height']
        head = float(prps['Heading'])
        try:
            gimy = float(prps['GimbalRollDegree'])
            print("Gimbal Roll Degree = %s" % str(gimy))
            gimx = float(prps['GimbalPitchDegree'])
            print("Gimbal Pitch Degree = %s" % str(gimx))
            gimz = float(prps['GimbalYawDegree'])
            print("Gimbal Yaw Degree = %s" % str(gimz))
        except:
            gimz = 0
        img_n = prps['File_Name']
        focal_lgth = prps['Focal_Length']
        alt = float(prps["Relative_Altitude"])
        # print(wid, hite, alt, focal_lgth, gimx, gimy, gimz, head)
        calc1, calc2 = get_area(wid, hite, alt, focal_lgth)
        #calc1 = calc1 / 1.87
        #calc2 = calc2 / 1.87
        # create geodataframe with some variables
        gf = gp.GeoDataFrame({'lat': lat, 'lon': lng, 'width': calc1, 'height': calc2}, index=[1])
        #gf = gp.GeoDataFrame({'lat': lat, 'lon': lng, 'width': calc2, 'height': calc1}, index=[1])
        repo = convert_wgs_to_utm(lng, lat)
        repo = '%g'%(float(repo))
        #gf.crs = fiona.crs.from_epsg(4326)

        # create center as a shapely geometry point type and set geometry of dataframe to this
        gf['center'] = gf.apply(lambda x: Point(x['lon'], x['lat']), axis=1)
        gf = gf.set_geometry('center')
        gf.crs = fiona.crs.from_epsg(4326)
        # change crs of dataframe to projected crs to enable use of distance for width/height
        # gf = gf.to_crs(epsg=repo)
        gf = gf.to_crs(epsg=3857)
        #gf = gf.to_crs(epsg=4978)
        # calculate bounding box distance between corners to fix proportions distorsion due to EPSG:3857
        #pnt1 = Point(80.99456, 7.86795)
        #pnt2 = Point(80.97454, 7.872174)
        #points_df = gp.GeoDataFrame({'geometry': [pnt1, pnt2]}, crs='EPSG:4326')
        #points_df = points_df.to_crs('EPSG:5234')
        #points_df2 = points_df.shift()  # We shift the dataframe by 1 to align pnt1 with pnt2
        #print(points_df.distance(points_df2))

        # calculate bounding box distance between corners to fix proportions distorsion due to EPSG:3857
        prescaled_poly = shapely.geometry.box(*gf['center'].buffer(1).total_bounds)
        x, y = prescaled_poly.exterior.coords.xy
        #print(x)
        #print(y)
        top_left = Point(x[1], y[1])
        top_right = Point(x[0], y[0])
        width_df = gp.GeoDataFrame({'geometry': [top_left, top_right]}, crs='EPSG:3857')
        #width_df = width_df.to_crs('EPSG:3857')
        width_df2 = width_df.shift()  # We shift the dataframe by 1 to align pnt1 with pnt2
        #print(width_df.distance(width_df2))

        #lower_left = Point(x[2], y[2])
        lower_right = Point(x[3], y[3])
        height_df = gp.GeoDataFrame({'geometry': [top_right, lower_right]}, crs='EPSG:3857')
        #height_df = height_df.to_crs('EPSG:3857')
        height_df2 = height_df.shift()
        #print(height_df.distance(height_df2))

        # Vincenty's Inverse method (source: https://nathanrooy.github.io/posts/2016-12-18/vincenty-formula-with-python/)
        # width scale factor
        width_df_vincenty = width_df.to_crs('EPSG:4326')
        width_df_vincenty['points'] = width_df_vincenty.apply(lambda x: [y for y in x['geometry'].coords], axis=1)
        width_vicenty_points = width_df_vincenty.to_dict('records')
        wth_pnt_topleft = [width_vicenty_points[0]['points'][0][1], width_vicenty_points[0]['points'][0][0]]
        wth_pnt_topleft.reverse()
        wth_pnt_topright = [width_vicenty_points[1]['points'][0][1], width_vicenty_points[1]['points'][0][0]]
        wth_pnt_topright.reverse()
        width_scaling_factor = vincenty_inverse(wth_pnt_topleft, wth_pnt_topright).m
        print("Width scaling factor: %f" % width_scaling_factor)
        # height scale factor
        height_df_vincenty = height_df.to_crs('EPSG:4326')
        height_df_vincenty['points'] = height_df_vincenty.apply(lambda x: [y for y in x['geometry'].coords], axis=1)
        height_vicenty_points = height_df_vincenty.to_dict('records')
        hgt_pnt_topright = [height_vicenty_points[0]['points'][0][1], height_vicenty_points[0]['points'][0][0]]
        hgt_pnt_topright.reverse()
        hgt_pnt_lowerright = [height_vicenty_points[1]['points'][0][1], height_vicenty_points[1]['points'][0][0]]
        hgt_pnt_lowerright.reverse()
        height_scaling_factor = vincenty_inverse(hgt_pnt_topright, hgt_pnt_lowerright).m
        print("Height scaling factor: %f" % height_scaling_factor)

        # create polygon using width and height
        gf['center'] = shapely.geometry.box(*gf['center'].buffer(1).total_bounds)
        #gf['polygon'] = gf.apply(lambda x: shapely.affinity.scale(x['center'], x['width']/width_scaling_factor, x['height']/height_scaling_factor), axis=1)
        gf['polygon'] = gf.apply(lambda x: shapely.affinity.scale(x['center'], x['height'] / width_scaling_factor,
                                                                  x['width'] / height_scaling_factor), axis=1)
        #gf['polygon'] = gf.apply(lambda x: shapely.affinity.scale(x['center'], x['width'],
        #                                                          x['height']), axis=1)
        #gf['polygon'] = gf['center']
        gf = gf.set_geometry('polygon')
        geopoly = gf['polygon'].to_json()
        g1 = geojson.loads(geopoly)
        gh = g1[0].geometry
        g2 = shape(gh)
        # Change Heading / orientation
        head_ck = 0
        if gimz > head_ck:
            header = 360 - gimz
        else:
            header = 360 - (gimz)
        # print(header)
        ngf = affinity.rotate(g2, header, origin='centroid')
        #print("NGF", ngf)
        over_poly.append(ngf)
        # therepo = 'epsg:' + repo
        # darepo = therepo
        project = partial(
            pyproj.transform,
            # pyproj.Proj(init=darepo),
            pyproj.Proj('epsg:3857'),  # source coordinate system
            pyproj.Proj('epsg:4326'))  # destination coordinate system
        g2 = transform(project, ngf)
        # Create GeoJSON
        wow3 = geojson.dumps(g2)
        wow4 = json.loads(wow3)
        wow4 = rewind(wow4)
        # propy = dict(image=img_n)
        # gd_feat = dict(Feature="Polygon", properties=prps, geometry=wow4)

        # pyGeom = dict(type="Polygon", coordinates=wow4)
        gd_feat = dict(type="Feature", geometry=wow4, properties=prps)
        polys.append(gd_feat)
        bar.next()
    union_buffered_poly = cascaded_union([l.buffer(.00001) for l in over_poly])
    poly = union_buffered_poly.simplify(10, preserve_topology=False)
    projected = partial(
        pyproj.transform,
        pyproj.Proj('epsg:3857'),  # source coordinate system
        #pyproj.Proj(init='epsg:4978'),  # source coordinate system
        pyproj.Proj('epsg:4326'))  # destination coordinate system
    g3 = transform(projected, poly)
    pop3 = geojson.dumps(g3)
    pop4 = json.loads(pop3)
    pop4 = rewind(pop4)
    bar.finish()
    return polys, pop4


def get_area(wd, ht, alt, fl):
    """
    :param wd:
    :param ht:
    :param alt:
    :param fl:
    :param gimx:
    :param gimy:
    :param gimz:
    :param head:
    :return:
    """
    #sw = 8 # INCORRECT Hasselblad L1D-20c
    #sh = 5.3 # INCORRECT Hasselblad L1D-20c
    #sw = 13.2 # Hasselblad L1D-20c (DJI FC6310 -> Phantom 4 PRO)
    #sh = 8.8 # Hasselblad L1D-20c (DJI FC6310 -> Phantom 4 PRO)
    sw = 7.68 # DJI ZH20T (640 x 12 [um]) / 1000 [um/mm] mm
    sh = 6.144 # DJI ZH20T (512 x 12 [um]) / 1000 [um/mm] mm
    #sw = 6.17 # DJI Phantom 4
    #sh = 4.55 # DJI Phantom 4
    #sw = 10.88 # DJI XT2 (640 x 17 [um]) / 1000 [um/mm] mm
    #sh = 8.704 # DJI XT2 (512 x 17 [um]) / 1000 [um/mm] mm

    xview = 2 * degrees(atan(sw / (2 * fl)))
    yview = 2 * degrees(atan(sh / (2 * fl)))
    newcl = calc_res(wd, ht, xview, yview, alt)

    # xground = alt * (tan(90 - gimx + 0.5 * xview)) - alt * (tan(90 - gimx - 0.5 * xview))
    # print(xground)
    gsd = (sw * alt * 100) / (fl * wd)
    grd_wdth = ((gsd * wd) / 100)
    grd_ht = ((gsd * ht) / 100)
    # print("with radians", newcl[1], newcl[2])
    # print("Without Radians", grd_wdth, grd_ht)
    # return grd_wdth, grd_ht
    return newcl[1], newcl[2]


def calc_res(pixel_x, pixel_y, x_angle, y_angle, alt):
    """
    :param pixel_x:
    :param pixel_y:
    :param x_angle:
    :param y_angle:
    :param alt:
    :param head:
    :param gimz:
    :return:
    """
    # setup vars

    # Calc IFOV, based on right angle trig of one-half the lens angles
    fov_x = 2 * (math.tan(math.radians(0.5 * x_angle)) * alt)
    fov_y = 2 * (math.tan(math.radians(0.5 * y_angle)) * alt)

    # Calc pixel resolutions based on the IFOV and sensor size
    pixel_resx = (fov_x / pixel_x) * 100
    pixel_resy = (fov_y / pixel_y) * 100

    # Average the X and Y resolutions
    pix_resolution_out = ((pixel_resx + pixel_resy) / 2)

    return pix_resolution_out, fov_x, fov_y


def convert_wgs_to_utm(lon, lat):
    """
    :param lon:
    :param lat:
    :return:
    """
    utm_band = str((math.floor((lon + 180) / 6) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = '0'+utm_band
    if lat >= 0:
        epsg_code = '326' + utm_band
    else:
        epsg_code = '327' + utm_band
    return epsg_code


def find_file(some_dir):
    """
    :param some_dir:
    :return:
    """
    matches = []
    for root, dirnames, filenames in os.walk(some_dir):
        for filename in fnmatch.filter(filenames, '*.JPG'):
            matches.append(os.path.join(root, filename))
    return matches


if __name__ == "__main__":
    main()
