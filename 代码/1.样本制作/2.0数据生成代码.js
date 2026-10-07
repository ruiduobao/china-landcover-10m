/**
 * GLC_FCS30D 批量分块采样脚本 (10x10度大网格 -> 1x1度分块导出)
 * * 流程：
 * 1. 定义一个 10x10 的大区域。
 * 2. 在全区域上应用 "7年稳定 + 形态学腐蚀" 算法。
 * 3. 循环遍历 100 个 1x1 的子网格。
 * 4. 为每个子网格生成一个独立的导出任务。
 */

// ================= 1. 参数设置 =================
// Function to recode class values into sequential values starting from 1 onwards
var recodeClasses = function(image) {
  // Define the class values
  var classes = [10, 11, 12, 20, 51, 52, 61, 62, 71, 72, 81, 82, 91, 92, 120, 121, 122, 
                 130, 140, 150, 152, 153, 181, 182, 183, 184, 185, 186, 187, 190, 200, 
                 201, 202, 210, 220, 0];
  var reclassed = image.remap(classes, ee.List.sequence(1, classes.length));
  return reclassed;
};

// Function to add a layer with given settings
var addLayer = function(image, name) {
  Map.addLayer(image, {palette: palette}, name,false);
};

// Apply the function to your images and add layers
addLayer(recodeClasses(five_year.mosaic().select('b1')), 'GLC FCS 1985');
addLayer(recodeClasses(five_year.mosaic().select('b2')), 'GLC FCS 1990',false);
addLayer(recodeClasses(five_year.mosaic().select('b3')), 'GLC FCS 1995',false);

// Load the GLC-FCS30D collection
var image = annual.mosaic();

// Iterate over each band (year) in the image
for (var i = 1; i <= 23; i++) {
  var year = 1999 + i; // starts at year 2000 for annual maps
  var layerName = "GLC FCS " + year.toString();
  var band = image.select("b" + i);
  
  // Apply the function to the band and add layer
  addLayer(recodeClasses(band), layerName);
}

// Define a dictionary for legend and visualization
var dict = {
  "names": [
    "Rainfed cropland",
    "Herbaceous cover cropland",
    "Tree or shrub cover (Orchard) cropland",
    "Irrigated cropland",
    "Open evergreen broadleaved forest",
    "Closed evergreen broadleaved forest",
    "Open deciduous broadleaved forest (0.15<fc<0.4)",
    "Closed deciduous broadleaved forest (fc>0.4)",
    "Open evergreen needle-leaved forest (0.15< fc <0.4)",
    "Closed evergreen needle-leaved forest (fc >0.4)",
    "Open deciduous needle-leaved forest (0.15< fc <0.4)",
    "Closed deciduous needle-leaved forest (fc >0.4)",
    "Open mixed leaf forest (broadleaved and needle-leaved)",
    "Closed mixed leaf forest (broadleaved and needle-leaved)",
    "Shrubland",
    "Evergreen shrubland",
    "Deciduous shrubland",
    "Grassland",
    "Lichens and mosses",
    "Sparse vegetation (fc<0.15)",
    "Sparse shrubland (fc<0.15)",
    "Sparse herbaceous (fc<0.15)",
    "Swamp",
    "Marsh",
    "Flooded flat",
    "Saline",
    "Mangrove",
    "Salt marsh",
    "Tidal flat",
    "Impervious surfaces",
    "Bare areas",
    "Consolidated bare areas",
    "Unconsolidated bare areas",
    "Water body",
    "Permanent ice and snow",
    "Filled value"
  ],
  "colors": [
    "#ffff64",
    "#ffff64",
    "#ffff00",
    "#aaf0f0",
    "#4c7300",
    "#006400",
    "#a8c800",
    "#00a000",
    "#005000",
    "#003c00",
    "#286400",
    "#285000",
    "#a0b432",
    "#788200",
    "#966400",
    "#964b00",
    "#966400",
    "#ffb432",
    "#ffdcd2",
    "#ffebaf",
    "#ffd278",
    "#ffebaf",
    "#00a884",
    "#73ffdf",
    "#9ebb3b",
    "#828282",
    "#f57ab6",
    "#66cdab",
    "#444f89",
    "#c31400",
    "#fff5d7",
    "#dcdcdc",
    "#fff5d7",
    "#0046c8",
    "#ffffff",
    "#ffffff",
    "#ffffff"
  ]
};


// 大网格起始点 (左下角)
var START_LON = 0; 
var START_LAT = 0;  

// 大网格跨度 (10度)
var TOTAL_GRID_SIZE = 10; 

// 子网格跨度 (1度，每个文件的大小)
var SUB_TILE_SIZE = 1;

// 每个 1x1 度网格内的采样目标数
// 注意：如果该网格是海洋或被过滤掉了，实际数量会少于此值
var SAMPLES_PER_SUB_TILE = 1000; 

// ================= 2. 数据准备与算法逻辑 (一次性定义) =================

// 为了避免循环内重复加载，先定义整个大区域的计算逻辑
var xMin_Big = START_LON;
var yMin_Big = START_LAT;
var xMax_Big = START_LON + TOTAL_GRID_SIZE;
var yMax_Big = START_LAT + TOTAL_GRID_SIZE;
var bigROI = ee.Geometry.Rectangle([xMin_Big, yMin_Big, xMax_Big, yMax_Big], 'EPSG:4326', false);

Map.centerObject(bigROI, 6);
Map.addLayer(bigROI, {color: 'red', fill: false}, 'Big Grid Boundary');

// --- 数据加载 ---
var collection = ee.ImageCollection("projects/sat-io/open-datasets/GLC-FCS30D/annual");
var glc_mosaic = collection.mosaic().clip(bigROI); // 先裁剪到大区域以减少计算

// --- 7年波段提取 (2016-2022) ---
var years = ee.List.sequence(2016, 2022);
var bandNames = years.map(function(y) {
  return ee.String('b').cat(ee.Number(y).subtract(2000).add(1).int());
});
var stack7Years = glc_mosaic.select(bandNames);

// --- 核心算法一：7年稳定性 ---
var minImg = stack7Years.reduce(ee.Reducer.min());
var maxImg = stack7Years.reduce(ee.Reducer.max());
var stableMask7Yr = minImg.eq(maxImg);

// --- 核心算法二：空间形态学腐蚀 (基于2020) ---
var lc2020 = glc_mosaic.select(['b21']).rename('class');
var kernel = ee.Kernel.circle({radius: 1.5, units: 'pixels'});
var spatialMin = lc2020.reduceNeighborhood({reducer: ee.Reducer.min(), kernel: kernel});
var spatialMax = lc2020.reduceNeighborhood({reducer: ee.Reducer.max(), kernel: kernel});
var morphMask = spatialMin.eq(spatialMax);

// --- 合并掩膜 ---
var finalMask = stableMask7Yr.and(morphMask);
var validData = lc2020.neq(0).and(lc2020.neq(250)); // 去除填充值
finalMask = finalMask.and(validData);

// --- 得到最终的"干净"影像 ---
// 注意：此时还没有开始计算，只是定义了图层
var cleanLC_Global = lc2020.updateMask(finalMask);


// ================= 3. 循环生成 100 个导出任务 =================

// 计算行列数 (例如 10 / 1 = 10 行 10 列)
var steps = TOTAL_GRID_SIZE / SUB_TILE_SIZE;

print('开始生成任务...');
print('总计将生成 ' + (steps * steps) + ' 个导出任务。');

// 客户端双重循环
for (var i = 0; i < steps; i++) {
  for (var j = 0; j < steps; j++) {
    
    // --- 3.1 计算当前子网格坐标 ---
    var currLon = START_LON + (i * SUB_TILE_SIZE);
    var currLat = START_LAT + (j * SUB_TILE_SIZE);
    
    // 构造当前子网格几何体
    var subTileGeom = ee.Geometry.Rectangle(
      [currLon, currLat, currLon + SUB_TILE_SIZE, currLat + SUB_TILE_SIZE], 
      'EPSG:4326', 
      false
    );
    
    // 构建唯一的 Task ID 字符串
    var tileID = 'Lon' + currLon + '_Lat' + currLat;
    
    // --- 3.2 针对当前子网格进行采样 ---
    // 即使 cleanLC_Global 是大的，stratifiedSample 会自动只处理 region 范围
    var samples = cleanLC_Global.stratifiedSample({
      numPoints: SAMPLES_PER_SUB_TILE,
      classBand: 'class',
      region: subTileGeom, // 关键：限制在当前 1x1 度网格内
      scale: 30,
      geometries: true,
      dropNulls: true,
      tileScale: 2 // 如果遇到内存溢出，将此值改为 4 或 8
    });
    
    // --- 3.3 数据清洗 (转纯文本) ---
    // 这里需要把 currLon 和 currLat 传进去，闭包陷阱注意：
    // 在 GEE 这种同步循环里，基础类型变量通常是安全的，但为了保险，直接在 map 里用 geometry 获取
    var finalSamples = samples.map(function(f) {
      var coords = f.geometry().coordinates();
      return f.set({
        'lon': coords.get(0),
        'lat': coords.get(1),
        'year': 2020,
        'tile_id': tileID // 将 Tile ID 写入每一行
      }).setGeometry(null); // 删除几何体减小体积
    });

    // --- 3.4 生成导出任务 ---
    // 任务名不能包含点号或特殊字符，转为整数或下划线
    var taskDesc = 'Sample_Lon' + currLon + '_Lat' + currLat;
    
    Export.table.toDrive({
      collection: finalSamples,
      description: taskDesc,
      folder: 'GLC_Samples_Grid_2x2', // 建议统一放在一个新文件夹
      fileFormat: 'CSV',
      selectors: ['lon', 'lat', 'class', 'year', 'tile_id']
    });
  }
}

print('任务生成完毕，请在右侧 Tasks 面板点击 Run。');
