import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.Pane;
// This exists to make a Pane with full size
public class ImagePane extends Pane {
	ImagePane(Image image){
		ImageView imageView = new ImageView(image);
		imageView.setFitWidth(DuckHunt.SCALE*256);
		imageView.setFitHeight(DuckHunt.SCALE*240);
		getChildren().add(imageView);
	}
}
