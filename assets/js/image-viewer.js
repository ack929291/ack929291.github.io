document.addEventListener("DOMContentLoaded", function () {
  var dialog = document.createElement("dialog");
  var closeButton = document.createElement("button");
  var enlargedImage = document.createElement("img");

  dialog.className = "image-lightbox";
  dialog.setAttribute("aria-label", "图片预览");

  closeButton.className = "image-lightbox__close";
  closeButton.type = "button";
  closeButton.setAttribute("aria-label", "关闭图片预览");
  closeButton.title = "关闭";
  closeButton.textContent = "×";

  enlargedImage.className = "image-lightbox__image";
  dialog.append(closeButton, enlargedImage);
  document.body.appendChild(dialog);

  document.querySelectorAll(".diagram-image__trigger").forEach(function (trigger) {
    trigger.addEventListener("click", function () {
      var image = trigger.querySelector("img");
      enlargedImage.src = image.currentSrc || image.src;
      enlargedImage.alt = image.alt;
      dialog.showModal();
    });
  });

  closeButton.addEventListener("click", function () {
    dialog.close();
  });

  dialog.addEventListener("click", function (event) {
    if (event.target === dialog) {
      dialog.close();
    }
  });
});
