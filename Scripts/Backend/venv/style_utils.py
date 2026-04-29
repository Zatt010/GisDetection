"""
Utility functions for creating styled vector exports with proper symbology
"""

import geopandas as gpd
import os
from datetime import datetime

# Class colors matching the original visualization
CLASS_COLORS = {
    0: '#006400',  # Bosque - Dark Green
    1: '#228B22',  # Matorrales - Forest Green  
    2: '#ADFF2F',  # Pastizales - Yellow Green
    3: '#FFFF00',  # T_Agricolas - Yellow
    4: '#FF0000',  # Infraestructura - Red
    5: '#8B4513',  # Suelo_Desnudo - Brown
    6: '#0000FF',  # Agua - Blue
}

CLASS_NAMES = ['Bosque', 'Matorrales', 'Pastizales', 'T_Agricolas', 'Infraestructura', 'Suelo_Desnudo', 'Agua']

def create_styled_geopackage(gdf, output_path):
    """
    Create a GeoPackage with embedded QML style file for proper visualization
    """
    # Add color attributes to GeoDataFrame
    gdf['fill_color'] = gdf['class_index'].map(CLASS_COLORS)
    gdf['stroke_color'] = gdf['class_index'].map(CLASS_COLORS)
    gdf['stroke_width'] = 0.5
    gdf['fill_opacity'] = 0.7
    
    # Save GeoPackage
    gdf.to_file(output_path, driver='GPKG')
    
    # Create QML style file for QGIS
    qml_path = output_path.replace('.gpkg', '.qml')
    create_qml_style(qml_path)
    
    return output_path, qml_path

def create_qml_style(output_path):
    """
    Create QML style file for QGIS that will auto-apply colors
    """
    qml_content = f'''<!DOCTYPE qml PUBLIC "http://www.qgis.org/qml" "http://www.qgis.org/qml.dtd">
<qml version="1.0">
  <pipe>
    <rasterrenderer opacity="0.7" alphaBand="-1" blueBand="1" greenBand="1" redBand="1" type="paletted">
      <rasterTransparency>
        <singleValuePixelList>
          <pixelListEntry color="#006400" label="Bosque" value="0"/>
          <pixelListEntry color="#228B22" label="Matorrales" value="1"/>
          <pixelListEntry color="#ADFF2F" label="Pastizales" value="2"/>
          <pixelListEntry color="#FFFF00" label="T_Agricolas" value="3"/>
          <pixelListEntry color="#FF0000" label="Infraestructura" value="4"/>
          <pixelListEntry color="#8B4513" label="Suelo_Desnudo" value="5"/>
          <pixelListEntry color="#0000FF" label="Agua" value="6"/>
        </singleValuePixelList>
      </rasterTransparency>
      <colorramp type="INTERPOLATED" >
        <colorrampshader color="#006400" label="Bosque" value="0"/>
        <colorrampshader color="#228B22" label="Matorrales" value="1"/>
        <colorrampshader color="#ADFF2F" label="Pastizales" value="2"/>
        <colorrampshader color="#FFFF00" label="T_Agricolas" value="3"/>
        <colorrampshader color="#FF0000" label="Infraestructura" value="4"/>
        <colorrampshader color="#8B4513" label="Suelo_Desnudo" value="5"/>
        <colorrampshader color="#0000FF" label="Agua" value="6"/>
      </colorramp>
    </rasterrenderer>
    <brightnesscontrast brightness="0" contrast="0"/>
    <huesaturation colorizeOn="0" colorizeRed="255" colorizeGreen="128" colorizeBlue="128" grayscaleMode="0" saturation="0"/>
  </pipe>
</qml>'''
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(qml_content)
    
    return output_path

def create_sld_style(output_path):
    """
    Create SLD style file for GeoServer/other GIS software
    """
    sld_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<StyledLayerDescriptor version="1.0.0" 
    xmlns="http://www.opengis.net/sld" 
    xmlns:ogc="http://www.opengis.net/ogc"
    xmlns:xlink="http://www.w3.org/1999/xlink"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="http://www.opengis.net/sld http://schemas.opengis.net/sld/1.0.0/StyledLayerDescriptor.xsd">
  <NamedLayer>
    <UserStyle>
      <FeatureTypeStyle>
        <Rule>
          <PolygonSymbolizer>
            <Fill>
              <CssParameter name="fill">#006400</CssParameter>
              <CssParameter name="fill-opacity">0.7</CssParameter>
            </Fill>
            <Stroke>
              <CssParameter name="stroke">#006400</CssParameter>
              <CssParameter name="stroke-width">0.5</CssParameter>
            </Stroke>
          </PolygonSymbolizer>
        </Rule>
        <Rule>
          <Filter xmlns:ogc="http://www.opengis.net/ogc">
            <ogc:PropertyIsEqualTo>
              <ogc:PropertyName>class_index</ogc:PropertyName>
              <ogc:Literal>1</ogc:Literal>
            </ogc:PropertyIsEqualTo>
          </Filter>
          <PolygonSymbolizer>
            <Fill>
              <CssParameter name="fill">#228B22</CssParameter>
              <CssParameter name="fill-opacity">0.7</CssParameter>
            </Fill>
            <Stroke>
              <CssParameter name="stroke">#228B22</CssParameter>
              <CssParameter name="stroke-width">0.5</CssParameter>
            </Stroke>
          </PolygonSymbolizer>
        </Rule>
      </FeatureTypeStyle>
    </UserStyle>
  </NamedLayer>
</StyledLayerDescriptor>'''
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(sld_content)
    
    return output_path

def create_legend_png(output_path):
    """
    Create a legend PNG file for reference
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        import matplotlib.pyplot as plt
        
        # Create legend image
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')
        
        # Create legend patches
        legend_elements = []
        for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS.values())):
            from matplotlib.patches import Patch
            legend_elements.append(Patch(facecolor=color, edgecolor='black', label=name))
        
        ax.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.5))
        plt.title('Leyenda de Clasificación', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    except ImportError:
        print("PIL/matplotlib not available for legend creation")
        return None
