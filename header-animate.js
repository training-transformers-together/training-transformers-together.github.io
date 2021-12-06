// draw background; Note: this background is based on https://codepen.io/pawelqcm/pen/oxPYox by Pawel
// Note 2: Pawel, you're awesome.
(function() {
  var content_element = document.getElementById("overlay");
  var canvas = document.querySelector('canvas');
  var title_elem = document.getElementsByClassName("faded title")[0];
  var title_text = document.getElementById("title_text");
  ctx = canvas.getContext('2d');
  if (!ctx)
      console.warn("Your browser does not support canvas, content may be broken :'(");

  var SENSITIVITY, SIBLINGS_LIMIT, DENSITY, TOTAL_NODES, ANCHOR_LENGTH, CURSOR_HEIGHT, CURSOR_WIDTH;
  css_opts = getComputedStyle(document.documentElement);
  SENSITIVITY = css_opts.getPropertyValue('--background-sensitivity') || 120;
  SIBLINGS_LIMIT = css_opts.getPropertyValue('--background-siblings') || 7;
  NODE_DENSITY = css_opts.getPropertyValue('--background-node-density') || 6;
  CURSOR_WIDTH = css_opts.getPropertyValue('--background-cursor-width') || 250;
  CURSOR_HEIGHT = css_opts.getPropertyValue('--background-cursor-height') || 250;
  CURSOR_VERTICAL_SHRINK = css_opts.getPropertyValue('--background-cursor-vertical-shrink') || 0.1;
  SPEED_COEF = css_opts.getPropertyValue('--background-speed') || 1;
  ENERGY_DECAY = css_opts.getPropertyValue('--energy-decay') || 2;
  SHOW_IF_WIDER_THAN = css_opts.getPropertyValue('--background-show-if-wider-than') || 500;
  MOVE_ON_CURSOR = css_opts.getPropertyValue('--background-move-on-cursor').includes("true") || false;

  var nodes = [];
  choice = (choices => choices[Math.floor(Math.random() * choices.length)])
  sample_color = () => choice([[40, 40, 40], [133, 133, 133]])

  ANCHOR_LENGTH = 20;

  var cursor = {x: 0, y: 0};

  function centralize_cursor() {
      var rect = document.getElementById("bug-logo").getBoundingClientRect()
      var window_left = window.pageXOffset || document.documentElement.scrollLeft;
      var window_top = window.pageYOffset || document.documentElement.scrollTop;
      cursor.x = window_left + rect.left + rect.width / 2;
      cursor.y = window_top + rect.top + rect.height / 2;
  }

  function Node(x, y) {
    this.anchorX = x;
    this.anchorY = y;
    this.x = Math.random() * (x - (x - ANCHOR_LENGTH)) + (x - ANCHOR_LENGTH);
    this.y = Math.random() * (y - (y - ANCHOR_LENGTH)) + (y - ANCHOR_LENGTH);
    this.vx = (Math.random() * 2 - 1) * SPEED_COEF;
    this.vy = (Math.random() * 2 - 1) * SPEED_COEF;
    this.energy = Math.random() * 100;
    this.radius = Math.random();
    this.siblings = [];
    [this.r, this.g, this.b] = sample_color()
    this.brightness = 0;
  }

  Node.prototype.drawNode = function() {
    var color = `rgba(${this.r}, ${this.g}, ${this.b}, ${this.brightness})`;
    ctx.beginPath();
    ctx.arc(this.x, this.y, 2 * this.radius + 2 * this.siblings.length / SIBLINGS_LIMIT, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
  };

  Node.prototype.drawConnections = function() {
    for (var i = 0; i < this.siblings.length; i++) {
      var color = `rgba(133, 133, 133, ${this.brightness})`;
      ctx.beginPath();
      ctx.moveTo(this.x, this.y);
      ctx.lineTo(this.siblings[i].x, this.siblings[i].y);
      ctx.lineWidth = 1 - calcDistance(this, this.siblings[i]) / SENSITIVITY;
      ctx.strokeStyle = color;
      ctx.stroke();
    }
  };


  Node.prototype.moveNode = function() {
    this.energy -= ENERGY_DECAY;
    if (this.energy < 1) {
      this.energy = Math.random() * 100;
      if (this.x - this.anchorX < -ANCHOR_LENGTH) {
        this.vx = Math.random() * SPEED_COEF;
      } else if (this.x - this.anchorX > ANCHOR_LENGTH) {
        this.vx = Math.random() * -SPEED_COEF;
      } else {
        this.vx = Math.random() * SPEED_COEF * 2 - SPEED_COEF;
      }
      if (this.y - this.anchorY < -ANCHOR_LENGTH) {
        this.vy = Math.random() * SPEED_COEF;
      } else if (this.y - this.anchorY > ANCHOR_LENGTH) {
        this.vy = Math.random() * -SPEED_COEF;
      } else {
        this.vy = Math.random() * SPEED_COEF * 2 - SPEED_COEF;
      }
    }
    relative_speed_rate = Math.min(canvas.height / 100, 10.0)
    this.x += this.vx * this.energy * relative_speed_rate;
    this.y += this.vy * this.energy * relative_speed_rate;
  };

  function initNodes() {
    centralize_cursor();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (canvas.width >= SHOW_IF_WIDER_THAN)
        total_nodes = Math.round(NODE_DENSITY * (canvas.width / 100 * canvas.height / 100));
    else
        total_nodes = 0;
    nodes = [];
    for (var i = 0; i < total_nodes; i++)
        nodes.push(new Node(50 + Math.random() * (canvas.width - 100),
                            5 + Math.random() * (canvas.height - 10)));
  }

  function calcDistance(node1, node2) {
    return Math.sqrt(Math.pow(node1.x - node2.x, 2) + (Math.pow(node1.y - node2.y, 2)));
  }

  function findSiblings() {
    var node1, node2, distance;
    for (var i = 0; i < nodes.length; i++) {
      node1 = nodes[i];
      node1.siblings = [];
      for (var j = 0; j < nodes.length; j++) {
        node2 = nodes[j];
        if (node1 !== node2) {
          distance = calcDistance(node1, node2);
          if (distance < SENSITIVITY) {
            if (node1.siblings.length < SIBLINGS_LIMIT) {
              node1.siblings.push(node2);
            } else {
              var node_sibling_distance = 0;
              var max_distance = 0;
              var s;
              for (var k = 0; k < SIBLINGS_LIMIT; k++) {
                node_sibling_distance = calcDistance(node1, node1.siblings[k]);
                if (node_sibling_distance > max_distance) {
                  max_distance = node_sibling_distance;
                  s = k;
                }
              }
              if (distance < max_distance) {
                node1.siblings.splice(s, 1);
                node1.siblings.push(node2);
              }
            }
          }
        }
      }
    }
  }

  function redrawScene() {
    resizeWindow();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    findSiblings();
    var i, node, distance;
    for (i = 0; i < nodes.length; i++) {
      node = nodes[i];
      scaled_distance = calcDistance({x: cursor.x / CURSOR_WIDTH, y: cursor.y / CURSOR_HEIGHT},
                                     {x: node.x / CURSOR_WIDTH, y: node.y / CURSOR_HEIGHT});

      node.brightness = Math.max(1 - scaled_distance, 0);
    }
    for (i = 0; i < nodes.length; i++) {
      node = nodes[i];
      if (node.brightness) {
        node.drawConnections();
        node.drawNode();
      }
      node.moveNode();
    }
    requestAnimationFrame(redrawScene);
  }

  function initHandlers() {
    document.addEventListener('resize', resizeWindow);
    document.addEventListener('orientationchange', resizeWindow);
    if (MOVE_ON_CURSOR) {
        document.addEventListener('mousemove', moveHandler);
        document.addEventListener('touchmove', moveHandler);
    }
  }

  function resizeWindow(evt) {
    var new_width, new_height;
    new_width = Math.round(Math.max(title_elem.getBoundingClientRect().right, window.innerWidth))
    if (screen.width < 640)
      title_text.style.fontSize = "24px";
    else
      title_text.style.fontSize = "32px";


    if (!MOVE_ON_CURSOR)
        new_height = Math.round(title_elem.getBoundingClientRect().top - canvas.getBoundingClientRect().top);
    else
        new_height = Math.round(Math.max(
            content_element.offsetHeight, content_element.scrollHeight,
            content_element.clientHeight, window.innerHeight));

    if (canvas.width != new_width || canvas.height != new_height) {
        canvas.width = new_width;
        canvas.height = new_height;
        initNodes();
    }
    if (!MOVE_ON_CURSOR)
        centralize_cursor();
  }

  function moveHandler(evt) {
    if (evt.type == "mousemove") {
        cursor.x = window.pageXOffset + evt.clientX;
        cursor.y = window.pageYOffset + evt.clientY;
    }
    else { // touch event
        cursor.x = window.pageXOffset + evt.changedTouches[0].clientX;
        cursor.y = window.pageYOffset + evt.changedTouches[0].clientY;
    }
  }

  initHandlers();
  initNodes();
  redrawScene();

})();