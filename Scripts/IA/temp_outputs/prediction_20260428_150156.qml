<!DOCTYPE qml PUBLIC "http://www.qgis.org/qml" "http://www.qgis.org/qml.dtd">
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
</qml>