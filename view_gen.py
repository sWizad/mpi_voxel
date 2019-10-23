def generateView(outputFolder, size, planes, f):
  fo = open("results/" + outputFolder + ".html", "w")
  with open("viewer/viewer_template.html", "r") as fi:
    lines = fi.readlines()
  for line in lines:
    if "{INJECTION}" in line:
      print(line)
      fo.write('var size = "' + size + '";\n')
      fo.write('var layerzero = "' + outputFolder + '/mpi$$$$.png";\n')
      fo.write('var num_layers = %d;\n' % len(planes))
    elif "{INJECTION2}" in line:
      fo.write("const focal_length_px = %f;\n" % f)
      fo.write("const pose_plane=%f;\n" % planes[0])
      fo.write("const planes = [" + ",".join([str(x) for x in planes]) + "];\n")
    else:
      fo.write(line)
  fo.close()


def generateWebGL(outputFile, w, h, planes,namelist, subplane, f, px, py):
  print("Generating WebGL viewer")
  fo = open(outputFile, "w")

  replacer = {}
  replacer["WIDTH"] = w;
  replacer["HEIGHT"] = h;
  replacer["SCALE"] = 1;
  replacer["PLANES"] = "[" + ",".join([str(x) for x in planes]) + "]"
  #replacer["nPLANES"] = len(planes);
  replacer["nSUBPLANES"] = subplane;
  replacer["F"] = f
  replacer["NAMES"] = namelist#"[\"\"]"
  replacer["PX"] = px
  replacer["PY"] = py

  with open("viewer/gl_template.html", "r") as fi:
    lines = fi.readlines()
  for line in lines:
    if "{INJECTION}" in line:
      print(line)
      st = """
    const w = {WIDTH};
    const h = {HEIGHT};
    const scale = {SCALE};

    const planes = {PLANES};
    const nSubPlanes = {nSUBPLANES};

    const f = {F} * scale;
    var names = {NAMES};
    var nMpis = names.length;

    const py = {PY} * scale;
    const px = {PX} * scale;
    """
      for k in replacer:
        st = st.replace("{" + k + "}", str(replacer[k]))

      fo.write(st + '\n')
    else:
      fo.write(line)
  fo.close()

# generateView("640,480", [2.3, 4.5], 2)
