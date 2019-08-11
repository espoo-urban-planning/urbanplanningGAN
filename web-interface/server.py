'''
import textwrap

sys.path.append('/usr/local/Cellar/nginx/1.15.2/html/n/'); import server; from server import *
def cb(self,foo):
 ce.delete(ce.getObjectsFrom(ce.scene))
 settings = DXFImportSettings()
 ce.importFile(foo, settings)

 graphlayer = ce.getObjectsFrom(ce.scene, ce.isGraphLayer, ce.withName("0"))[0]
 segments = ce.getObjectsFrom(graphlayer, ce.isGraphSegment)
 settings = SimplifyGraphSettings()
 settings.setThresholdAngle(10)
 ce.simplifyGraph(segments, settings)

 cleanupSettings = CleanupGraphSettings()
 cleanupSettings.setIntersectSegments(True)
 cleanupSettings.setMergeNodes(True)
 cleanupSettings.setMergingDist(10)
 cleanupSettings.setSnapNodesToSegments(True)
 cleanupSettings.setSnappingDist(10)
 cleanupSettings.setResolveConflictShapes(True)
 graphlayer = ce.getObjectsFrom(ce.scene, ce.isGraphLayer)
 ce.cleanupGraph(graphlayer, cleanupSettings)

 views = ce.getObjectsFrom(ce.get3DViews())
 views[0].frame()
 ce.waitForUIIdle()
 png_filename = foo.replace(".dxf", ".png")
 png_filename = foo.replace(".dxf", "_out.png")
 views[0].snapshot(png_filename,width=1024,height=1024)


formatted=textwrap.dedent(code)
exec(formatted)


'''


from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
import SocketServer
import os.path
import os
import time
import cgi
import subprocess
from time import sleep
from random import randint
import logging as l
import base64

#settings = DXFImportSettings()
#ce.importFile("/usr/local/Cellar/nginx/1.15.2/html/n/tmp/out.dxf", settings)

current_milli_time = lambda: int(round(time.time() * 1000))
work_dir = '/usr/local/Cellar/nginx/1.15.2/html/n/'
save_path = '/usr/local/Cellar/nginx/1.15.2/html/n/tmp/'
log_file = work_dir + "server.log"

l.basicConfig(filename=log_file, level=l.INFO, filemode='w', format='%(name)s - %(levelname)s - %(message)s')

class S(BaseHTTPRequestHandler):
    callback = None
    
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

    def do_GET(self):
        self._set_headers()
        self.wfile.write("<html><body><h1>hi!</h1></body></html>")

    def do_HEAD(self):
        self._set_headers()
    
    def do_POST(self):
        l.debug('got POST request')
        self._set_headers()
        


        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD':'POST','CONTENT_TYPE':self.headers['Content-Type'],})


        data_raw_b64 = form["file_b64"].value
        l.info("got file" + str(data_raw_b64))
        data_raw_b64_cleaned = data_raw_b64.replace("data:image/png;base64,", "", 1) 
        l.info("attemptin to decode: " + str(data_raw_b64_cleaned))
        data_decoded = base64.b64decode(data_raw_b64_cleaned)

        base_filename = str(current_milli_time()) 
        outfilename = base_filename + ".png"

        fo = open(save_path + outfilename, "wb")
        fo.write(data_decoded)
        fo.close()

        cmnd = "/usr/local/bin/python /usr/local/Cellar/nginx/1.15.2/html/n/process.py " + outfilename
        l.info("attempting to run command " + cmnd)
        sleep(2)

        try:
            output = subprocess.check_output(cmnd, shell=True)
        except subprocess.CalledProcessError as e:
            output = "Error: " + str(e.output)

        

        l.info('attempting to callback')
        self.callback("/usr/local/Cellar/nginx/1.15.2/html/n/tmp/" + base_filename + ".dxf")
        self.wfile.write(base_filename)
        l.info('finished callback line')

        #os.system(cmnd)
        #process.process(outfilename)
        
def run(server_class=HTTPServer, handler_class=S, port=None, _callback=None):
    if(port == None):
        port = randint(5000, 15000)

    l.info('Starting server on port ' + str(port))
    server_address = ('', port)
    S.callback = _callback

    httpd = server_class(server_address, handler_class)

    l.info('Starting httpd...')
    httpd.serve_forever()

if __name__ == "__main__":
#from sys import argv

#if len(argv) == 2:
#    run(port=int(argv[1]))
#else:
    random_port = randint(5000, 15000)
    run(port=random_port)
