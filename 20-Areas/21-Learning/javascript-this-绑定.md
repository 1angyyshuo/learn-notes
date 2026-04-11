---
title: JavaScript this 绑定
date: 2024-04-11
created: 2024-04-11 20:00
tags: [javascript, 核心概念]
category: 前端开发
status: 进行中
---

# JavaScript this 绑定

## 概述

`this` 是 JavaScript 中最容易让人困惑的关键字之一。它的值取决于**函数的调用方式**，而不是定义位置。

## this 绑定的四条规则

### 1. 默认绑定

```javascript
function foo() {
  console.log(this);  // 严格模式下 undefined，非严格模式 window/global
}

foo();  // 默认绑定
```

### 2. 隐式绑定

```javascript
const obj = {
  name: '张三',
  sayHi: function() {
    console.log(this.name);
  }
};

obj.sayHi();  // "张三" - this 指向 obj
```

### 3. 显式绑定

```javascript
function sayName() {
  console.log(this.name);
}

const person = { name: '李四' };

sayName.call(person);   // "李四"
sayName.apply(person);  // "李四"
sayName.bind(person)(); // "李四"
```

### 4. new 绑定

```javascript
function Person(name) {
  this.name = name;
}

const p = new Person('王五');
console.log(p.name);  // "王五"
```

## 箭头函数的 this

箭头函数没有自己的 `this`，它会继承外层作用域的 `this`：

```javascript
const obj = {
  name: '测试',
  regularFunc: function() {
    console.log(this.name);  // "测试"
    
    setTimeout(function() {
      console.log(this.name);  // undefined（this 丢失）
    }, 100);
    
    setTimeout(() => {
      console.log(this.name);  // "测试"（继承外层 this）
    }, 100);
  }
};
```

## 关联笔记

- [[javascript-closure|JavaScript 闭包]] - 闭包与 this 常常一起考察
- [[javascript-原型链]] - 完整的 JS 核心概念

## 参考资料

- [MDN - this](https://developer.mozilla.org/zh-CN/docs/Web/JavaScript/Reference/Operators/this)
